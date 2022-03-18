from transformers import BertTokenizer
from typing import List, Optional


class BertEntityMarkerTokenizer(BertTokenizer):
    """
    Annotating Entity Markers
    """

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


class BertPEEMTokenizer(BertEntityMarkerTokenizer):
    """
    Annotating Positional embeddings + Entity Markers
    """

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed based on their position of embeddings to be used in a sequence-pair
        classification task. this sequence
        pair mask has the following format:

        ```
        |..<e1>...</e1>.....<e2>......</e2>...|..<e1>...</e1>.....<e2>......</e2>...|
        0 0 2 2 2 2 2 0 0 0 3 3 3 3 3 3 3 0 0 1 1 2 2 2 2 2 1 1 1 1 3 3 3 3 3 3 1 1
        |        first sequence              |          second sequence            |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        def create_token_type_ids(token_ids, e1_start: int, e1_end: int, e2_start: int,
                                  e2_end: int, base_state=0) -> List[int]:
            token_type_ids = []
            state = base_state
            for token_id in token_ids:
                if token_id == e1_start:  # seen token <e1>
                    assert state == 0, 'there must be no tags before "<e1>"'
                    state = 2
                    token_type_ids.append(state)
                elif token_id == e1_end:  # seen token </e1>
                    assert state == 2, '"<e1>" tag should be before "</e1>"'
                    token_type_ids.append(state)
                    state = 0
                elif token_id == e2_start:  # seen token <e2>
                    assert state == 0, 'no tags should be open when processing "<e2>"'
                    state = 3
                    token_type_ids.append(state)
                elif token_id == e2_end:  # seen token <e2/>
                    assert state == 3, '"<e2>" tag should be before "</e2>"'
                    token_type_ids.append(state)
                    state = 0
                else:
                    token_type_ids.append(state)
            return token_type_ids

        e1_start_token_id = self.convert_tokens_to_ids("<e1>")
        e1_end_token_id = self.convert_tokens_to_ids("</e1>")
        e2_start_token_id = self.convert_tokens_to_ids("<e2>")
        e2_end_token_id = self.convert_tokens_to_ids("</e2>")
        if token_ids_1 is None:
            # [CLS] + token_type_ids for sequence 0 + [SEP]
            return [0] + create_token_type_ids(token_ids_0, e1_start_token_id, e1_end_token_id, e2_start_token_id,
                                               e2_end_token_id, base_state=0) + [0]
        # [CLS] + token_type_ids for sequence 0 + [SEP] + token_type_ids for sequence 1 + [SEP]
        return [0] + create_token_type_ids(token_ids_0, e1_start_token_id, e1_end_token_id, e2_start_token_id,
                                           e2_end_token_id, base_state=0) + [0] + create_token_type_ids(token_ids_1,
                                                                                                        e1_start_token_id,
                                                                                                        e1_end_token_id,
                                                                                                        e2_start_token_id,
                                                                                                        e2_end_token_id,
                                                                                                        base_state=1)
