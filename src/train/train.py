from src.train.train_state import build_arg_parser, evaluate_model_state, main


if __name__ == "__main__":
    parser = build_arg_parser(task_default="soc", include_task=False)
    args = parser.parse_args()
    args.task = "soc"
    args.use_pos_enc = not args.disable_pos_enc
    args.use_token_embed = not args.disable_token_embed
    args.use_freq_gate = not args.disable_freq_gate
    main(args)
