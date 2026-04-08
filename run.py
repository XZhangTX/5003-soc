from src.train.train import build_arg_parser, main


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
