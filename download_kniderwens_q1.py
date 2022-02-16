from optparse import OptionParser

import os
import errno
import tarfile
import pandas as pd

from torchvision.datasets.utils import download_url

import file_handling as fh
from langdetect import detect
from sklearn.model_selection import train_test_split

class Kinderwens:

    """`LISS Panel Kinderwens 2021 <https://www.lissdata.nl/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, load the training data, otherwise test
        strip_html (bool, optional): If True, remove html tags during preprocessing; default=True
    """
    raw_filename = 'L_Kinderwens_2021_tscan_w_nonres_rt.csv'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    unlabeled_file = 'unlabeled.jsonlist'
    all_file = 'all.jsonlist'

    def __init__(self, root):
        super().__init__()
        self.root = os.path.expanduser(root)

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found.')

        self.preprocess()

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.unlabeled_file))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_filename))

    def preprocess(self):
        """Preprocess the raw data file"""
        if self._check_processed_exists():
            return

        train_lines = []
        test_lines = []
        unlabeled_lines = []
        all_lines = []

        POSITIVE_SET = set([
            'Zeker wel',
            'Waarschijnlijk wel',
            ])

        CERTAINTY_SET = set([
            'Zeker wel',
            'Zeker niet',
            ])

        print("Opening tar file")
        # read in the raw data
        dta = pd.read_csv(os.path.join(self.root, self.raw_filename))
        # process all the data in the archive
        print("Processing documents")
        for m_i, member in dta.iterrows():
            # Display occassional progress
            if (m_i + 1) % 50 == 0:
                print("Processed {:d} / {:d}".format(m_i+1, dta.shape[0]))

            if member['q1_Word_per_doc'] > 0 and member['nonresponse1'] != 0:
                doc = {
                    'id': member['nomem_encr'],
                    'text': member['FirstOpen'],
                    'intention': member['Intentie1'],
                    'positivity': member['Intentie1'] in POSITIVE_SET,
                    'certainty': member['Intentie1'] in CERTAINTY_SET,
                    'maternity': member['Zelf_Kind'] == 'Ja',
                    }

                if detect(doc['text']) in ['nl', 'af']:
                    all_lines.append(doc)
                else:
                    print(doc['text'])

        train_lines, test_lines = train_test_split(all_lines)
        print("Saving processed data to {:s}".format(self.root))
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.test_file))
        # fh.write_jsonlist(unlabeled_lines, os.path.join(self.root, self.unlabeled_file))
        fh.write_jsonlist(all_lines, os.path.join(self.root, self.all_file))


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', type=str, default='./data/kinderwens_q1',
                      help='Destination directory: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    root_dir = options.root_dir
    Kinderwens(root_dir)


if __name__ == '__main__':
    main()
