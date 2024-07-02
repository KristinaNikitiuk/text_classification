import os
import pandas as pd
from glob import glob

os.system("du -a /opt/ml")

SRCTRAINFILE = glob("/opt/ml/processing/input_train/*.csv")[0]
SRCTESTFILE = glob("/opt/ml/processing/input_test/*.csv")[0]

DSTTRAINFILE = "/opt/ml/processing/train/train.csv"
DSTTESTFILE = "/opt/ml/processing/test/test.csv"


class InputDataProcessing:
    def __init__(self):
        pass

    def _read_csv(self) -> (pd.DataFrame, pd.DataFrame):
        """
        load csv files from source directories
        :return: train and test dataframes
        """
        trainFrame = pd.read_csv(SRCTRAINFILE, header=0, sep=';')
        testFrame = pd.read_csv(SRCTESTFILE, header=0, sep=';')
        return trainFrame, testFrame

    def _text_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df: input dataframe
        :return: dataframe with updated text
        """
        df['txt'] = df['txt'].str.replace(",", "&#44;")
        return df

    def _save_to_csv(self, df: pd.DataFrame, result_file_path: str) -> None:
        """
        :param df: updated dataframe
        :param result_file_path: path to save dataframe
        """
        df.to_csv(
            path_or_buf=result_file_path,
            header=False,
            index=False,
            escapechar="\\",
            doublequote=False,
            quotechar='"',
        )

    def preprocess_data_flow(self):
        """
            preprocess and clean up input data
        """
        trainFrame, testFrame = self._read_csv()
        trainFrame = self._text_cleanup(trainFrame)
        testFrame = self._text_cleanup(testFrame)
        self._save_to_csv(df=trainFrame, result_file_path=DSTTRAINFILE)
        self._save_to_csv(df=testFrame, result_file_path=DSTTESTFILE)


if __name__ == "__main__":
    InputDataProcessing().preprocess_data_flow()
