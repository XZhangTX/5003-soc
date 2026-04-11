from src.train.train_state import build_arg_parser, evaluate_model_state, main


if __name__ == "__main__":
    parser = build_arg_parser(task_default="soc", include_task=False)
    args = parser.parse_args()
    args.task = "soc"
    main(args)
