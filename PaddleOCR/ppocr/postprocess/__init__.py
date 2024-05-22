# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy

__all__ = ["build_post_process"]

from .cls_postprocess import ClsPostProcess
from .db_postprocess import DBPostProcess, DistillationDBPostProcess
from .east_postprocess import EASTPostProcess
from .fce_postprocess import FCEPostProcess
from .pg_postprocess import PGPostProcess
from .rec_postprocess import (
    AttnLabelDecode,
    CTCLabelDecode,
    DistillationCTCLabelDecode,
    NRTRLabelDecode,
    PRENLabelDecode,
    SARLabelDecode,
    SEEDLabelDecode,
    SRNLabelDecode,
    TableLabelDecode,
)
from .sast_postprocess import SASTPostProcess
from .vqa_token_re_layoutlm_postprocess import VQAReTokenLayoutLMPostProcess
from .vqa_token_ser_layoutlm_postprocess import VQASerTokenLayoutLMPostProcess


def build_post_process(config, global_config=None):
    support_dict = [
        "DBPostProcess",
        "EASTPostProcess",
        "SASTPostProcess",
        "FCEPostProcess",
        "CTCLabelDecode",
        "AttnLabelDecode",
        "ClsPostProcess",
        "SRNLabelDecode",
        "PGPostProcess",
        "DistillationCTCLabelDecode",
        "TableLabelDecode",
        "DistillationDBPostProcess",
        "NRTRLabelDecode",
        "SARLabelDecode",
        "SEEDLabelDecode",
        "VQASerTokenLayoutLMPostProcess",
        "VQAReTokenLayoutLMPostProcess",
        "PRENLabelDecode",
        "DistillationSARLabelDecode",
    ]

    if config["name"] == "PSEPostProcess":
        from .pse_postprocess import PSEPostProcess

        support_dict.append("PSEPostProcess")

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        f"post process only support {support_dict}"
    )
    module_class = eval(module_name)(**config)
    return module_class
