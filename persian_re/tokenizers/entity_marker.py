from transformers import BertTokenizer


class BertEntityMarkerTokenizer(BertTokenizer):
    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
                 strip_accents=None,
                 **kwargs):
        super(BertEntityMarkerTokenizer, self).__init__(
            vocab_file,
            do_lower_case,
            do_basic_tokenize,
            never_split,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
            **kwargs
        )
        # todo: submit PR: missing typehint for
        #  additional_special_tokens values which accepts list or tuple but it's not type hinted
        added_tokens_num: int = self.add_special_tokens(
            {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})
