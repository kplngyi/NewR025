from eegbci_common import build_eegbci_parser, run_eegbci_experiment


if __name__ == "__main__":
    args = build_eegbci_parser(default_model="deep4")
    run_eegbci_experiment(args)
