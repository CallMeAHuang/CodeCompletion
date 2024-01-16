from beam import Beam
from dataset import LineDataset
from fuzzywuzzy import fuzz
from clearml import Task, Logger
from utils import *
import pandas
from knn_build import *

logger = logging.getLogger(__name__)



def eval_line_completion(args, model, tokenizer, file_type='test', load_file="train", res_file="dense.pkl"):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommanded to use single GPU because it could not be batched.
    """

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")
    
    dataset = LineDataset(tokenizer, args, logger, file_type=file_type, block_size=1024 - 100, 
                          cand_block_size = args.max_chunk_len * 2, load_file=load_file, search_res=res_file)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    model.eval()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    if args.langs == "python":
        break_ids = [tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'),
                     tokenizer.convert_tokens_to_ids('Ġ{')]
    preds = []
    gts = []
    edit_sim = 0.0
    em = 0.0
    dimension = model.config.hidden_size
    start_time = time.time()
    db_search_time = 0
    for step, (inputs, cands, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        cands = cands.squeeze(dim=0).to(args.device)
        with torch.no_grad():
            # 1. 计算数据库
            db_start_time = time.time()
            knn_saver = KNNSaver(dimension=dimension, pad_id=tokenizer.pad_token_id,
                                 only_errors=args.only_errors)
            knn_saver.break_into(model)
            model(cands, labels=cands)
            dstore_keys = knn_saver.dstore_keys
            dstore_vals = knn_saver.dstore_vals
            knn_saver.break_out()
            del knn_saver
            db_end_time = time.time()
            db_search_time += db_end_time - db_start_time
            # 2. 计算先验值
            knn_wrapper = KNNWrapper(dimension=dimension, keys=dstore_keys, vals=dstore_vals,
                                     use_knn=args.use_knn, lmbda=args.lmbda,
                                     use_knm=args.use_knm, use_bayes=args.use_bayes, window_size=args.window_size,
                                     knn_method=args.knn_method, pad_id=tokenizer.pad_token_id, k=args.token_k)
            knn_wrapper.break_into(model)
            beam_size = 5
            m = torch.nn.LogSoftmax(dim=-1)
            model_outputs = model(inputs[:, :-1])
            outputs = model_outputs[1]
            p = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            # 3. beam search
            for i in range(inputs.shape[0]):
                if args.model_type == "rnn":
                    past_hidden = tuple(x[:, i:i + 1].expand(-1, beam_size, -1).contiguous() for x in outputs)
                else:
                    past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                            outputs]
                    past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)
                input_ids = None
                for _ in range(100):
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    if args.model_type == "rnn":
                        outputs = model(input_ids, hidden=repackage_hidden(past_hidden))
                    else:
                        outputs = model(input_ids, past_key_values=past_hidden)
                    if knn_wrapper is not None:  # already done softmax
                        out = torch.log(outputs[0][:, -1, :]).data
                    else:
                        out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    if args.model_type == "rnn":
                        past_hidden = tuple(
                            x.data.index_select(1, beam.getCurrentOrigin()).contiguous() for x in outputs[1])
                    else:
                        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for
                                x in outputs[1]]
                        past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = DecodeIds(t).strip("<EOL>").strip()
                else:
                    text = DecodeIds(t).strip("{").strip()
                #print(text)
                # exit()
                preds.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0
            knn_wrapper.break_out()
            del knn_wrapper

        if step % 100 == 0:
            logger.info(f"{step} are done!")
            logger.info(f"Edit sim: {edit_sim / len(preds)}, EM: {em / len(preds)}")
            all_time = time.time() - start_time
            logger.info(f"step: {step}, time: {all_time}, db time: {db_search_time}, search time: {all_time - db_search_time}")

    file_name = "prediction_line_reacc.txt"
    saved_file = os.path.join(args.output_dir, file_name)
    with open(saved_file, "w") as f:
        for i, (pred_text, gt) in enumerate(zip(preds, gts)):
            if pred_text == gt:
                label = 1
            else:
                label = 0
            save_json = {
                'label': label,
                'pred': pred_text,
                'gt': gt
            }

            f.write(json.dumps(save_json) + "\n")

    logger.info(f"Test {len(preds)} samples")
    logger.info(f"Edit sim: {edit_sim / len(preds)}, EM: {em / len(preds)}")

    result = {
        "Edit": float(edit_sim / len(preds)),
        "EM": float(em / len(preds))
    }

    output_eval_file = os.path.join(args.output_dir, "eval_line_result.txt")
    with open(output_eval_file, "w") as writer:
        # logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            # logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


def add_args(parser):
    # 数据库相关
    parser.add_argument("--dstore_file", default=None, type=str, required=True,
                        help="The datastore file path. [domain training file]")

    # 预训练模型相关
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pretrain_dir_retr", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")

    # 待补全文件相关
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    # 检索相关
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_chunk_len", default=300, type=int)
    parser.add_argument("--block_k", default=10, type=int)
    parser.add_argument("--token_k", default=20, type=int)
    parser.add_argument("--use_knn", action="store_true")
    parser.add_argument("--lmbda", default=0.1, type=float)
    parser.add_argument("--only_errors", action="store_true")
    parser.add_argument("--use_knm", action="store_true")
    parser.add_argument("--use_bayes", action="store_true")
    parser.add_argument("--window_size",default=8, type=int)
    parser.add_argument('--knn_method', type=str, default='original')
    
    # 命令相关
    parser.add_argument("--data_process", action="store_true")
    parser.add_argument("--build_index", action='store_true')
    parser.add_argument("--do_search", action='store_true')
    parser.add_argument("--do_generate", action='store_true')
    parser.add_argument("--use_dense", action='store_true')
    parser.add_argument("--use_bm25", action='store_true')
    parser.add_argument("--use_hybrid", action='store_true')
    parser.add_argument("--bm_name", default="bm25", type=str, required=False,
                        help="elasticsearch name.")
    parser.add_argument('--clearml_proj_name', type=str, default='Hybrid')
    parser.add_argument('--task_name', type=str, default='')
    parser.add_argument('--log_file', type=str, default='log.log')

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.bm_name = args.model_type.lower()  # elasticsearch needs lower index
    args.dstore_path = args.output_dir + '/datastore'
    print(args)
    print(args.dstore_path)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # 为了测试需要，需要删除所有文件
    if not os.path.exists(args.dstore_path):
        os.makedirs(args.dstore_path)

    description = args.pretrain_dir

    if args.use_bm25:
        description += "__bm25"
    if args.use_hybrid:
        description += "__hybrid"
    if args.use_dense:
        description += "__dense"

    Task.init(project_name=args.clearml_proj_name, task_name=args.task_name)

    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)


    # 加载generator
    generator_tokenizer, generator_model = load_pretrained_model(args, special_tokens)

    if args.data_process:
        print("<!--- process train dataset --->")
        # 1. 为数据库切分代码
        split_file_path = split_code(args.dstore_file, args.dstore_path, max_chunk_len=args.max_chunk_len)
        # 3. 为token completion 处理数据
        #build_token_completion_data(generator_tokenizer, args, logger, file_type='test', block_size=1024)
    else:
        split_file_path = args.dstore_path + '/' + args.dstore_file.split("/")[-1].split(".")[0] + "_split.txt"

    before_contexts_file = os.path.join(args.data_dir, "test.json")

    # time 包含三部分: 搜索 + 推断
    time_search = 0
    if args.do_search:
        start_time = time.time()
        if args.use_bm25:
            logger.info('<!-- do bm25 search -->')
            tmp_dir = args.dstore_path + "/tmp"
            file_path = args.dstore_path + "/bm25.pkl"
            search_bm25(split_file_path, before_contexts_file, tmp_dir, args.bm_name, file_path, max_chunk_len=args.max_chunk_len)
            save_path = args.dstore_path + "/bm25_res.pkl"
            get_res(bm25_file=file_path, dense_file="", save_file=save_path, alpha=0.9, k=args.block_k)
        end_time = time.time()
        time_search += end_time - start_time
    
    if args.do_generate:
        logger.info('<!-- do generate -->')
        load_file = "train_split" if args.use_dense or args.use_bm25 else None
        if args.use_bm25:
            res_file = "bm25_res"
        else:
            res_file = None

        result = eval_line_completion(args, generator_model, generator_tokenizer, file_type='test',
                                      load_file=load_file, res_file=res_file)

        print(result)


if __name__ == '__main__':
    main()