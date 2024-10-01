import torch.nn as nn
from core.utils import (
    add_task_specific,
    add_aiov2_decoder_specific,
    add_aiov2_backbone_specific,
    add_aiov2_modality_specific,
)


class aio_entry_v2mae_shareneck(nn.Module):
    def __init__(
        self,
        backbone_module,
        patch_neck_module,
        label_neck_module,
        decoder_module,
        patch_adapter_module=None,
        label_adapter_module=None,
        patch_proj_module=None,
        label_proj_module=None,
        modalities={},
        kwargs={},
    ):
        super().__init__()
        self.backbone_module = backbone_module
        self.decoder_module = decoder_module
        self.modalities = modalities
        self.kwargs = kwargs
        self.test_flag = self.kwargs.get("test_flag", None)
        self.flip_channels = self.kwargs.get("flip_channels", False)

        self.add_module(
            "_".join(["adapter", self.modalities["patch"]]), patch_adapter_module
        )
        self.add_module(
            "_".join(["adapter", self.modalities["label"]]), label_adapter_module
        )

        patch_adatper_name = "self.adapter_{}".format(self.modalities["patch"])
        label_adatper_name = "self.adapter_{}".format(self.modalities["label"])

        self.add_module("_".join(["neck", "patch"]), patch_neck_module)
        self.add_module("_".join(["neck", "label"]), label_neck_module)

        patch_neck_name = "self.neck_patch"
        label_neck_name = "self.neck_label"

        self.add_module("_".join(["proj", self.modalities["patch"]]), patch_proj_module)
        self.add_module("_".join(["proj", self.modalities["label"]]), label_proj_module)

        patch_proj_name = "self.proj_{}".format(self.modalities["patch"])
        label_proj_name = "self.proj_{}".format(self.modalities["label"])

        self.patch_adatper_name = patch_adatper_name
        self.label_adapter_name = label_adatper_name
        self.patch_neck_name = patch_neck_name
        self.label_neck_name = label_neck_name
        self.patch_proj_name = patch_proj_name
        self.label_proj_name = label_proj_name

        add_task_specific(self, False)

        # as using the add_module in nn.Module(), the module names are feasible,
        # here we use the eval() with the module name to represent the
        # "self.neck_rgb" module with eval("self.neck_rgb")

        # modality share is truly the task share, e.g., all pose datasets share a same task,
        # therefore, the modality shared parameters are used as the task tokens.
        add_aiov2_modality_specific(
            eval(patch_adatper_name),
            self.modalities["patch"],
            True,
            eval(patch_adatper_name).task_sp_list,
            eval(patch_adatper_name).modality_share_list,
        )

        add_aiov2_modality_specific(
            eval(label_adatper_name),
            self.modalities["label"],
            True,
            eval(label_adatper_name).task_sp_list,
            eval(patch_adatper_name).modality_share_list,
        )

        add_aiov2_modality_specific(
            eval(patch_proj_name),
            self.modalities["patch"],
            True,
            eval(patch_proj_name).task_sp_list,
            eval(patch_proj_name).modality_share_list,
        )

        add_aiov2_modality_specific(
            eval(label_proj_name),
            self.modalities["label"],
            True,
            eval(label_proj_name).task_sp_list,
            eval(label_proj_name).modality_share_list,
        )

        add_aiov2_backbone_specific(
            self.backbone_module,
            True,
            self.backbone_module.task_sp_list,
            self.backbone_module.neck_sp_list,
        )
        add_aiov2_decoder_specific(
            self.decoder_module,
            True,
            self.decoder_module.task_sp_list,
            self.decoder_module.neck_sp_list,
            self.decoder_module.modality_share_list,
        )

        #  setting the neck as the same as the backbone (all shared parameters)
        add_aiov2_decoder_specific(
            eval(patch_neck_name),
            True,
            self.backbone_module.task_sp_list,
            self.backbone_module.neck_sp_list,
        )
        add_aiov2_decoder_specific(
            eval(label_neck_name),
            True,
            self.backbone_module.task_sp_list,
            self.backbone_module.neck_sp_list,
        )

    def forward(self, input_var, current_step):
        if self.training:
            input_var = eval(self.patch_adatper_name)(
                input_var
            )  # add key "patch tokens" to the dict
            input_var = eval(self.label_adapter_name)(
                input_var
            )  # add key "label tokens" to the dict
            x = self.backbone_module(
                input_var
            )  # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
            x = eval(self.patch_neck_name)(x)
            x = eval(self.label_neck_name)(x)

            decoder_feature = self.decoder_module(x)
            patch_outputs = eval(self.patch_proj_name)(decoder_feature)
            # import pdb;pdb.set_trace()
            label_outputs = eval(self.label_proj_name)(decoder_feature)
            output = {}
            output["outputs"] = patch_outputs
            output["outputs"].update(label_outputs)
        else:
            # task_flag
            if self.test_flag is None:
                output = self.forward_default_test(input_var, current_step)

            else:
                raise ValueError(
                    "test_flag ({}) is NOT supported!".format(self.test_flag)
                )

        return output

    def forward_default_test(self, input_var, current_step):
        # input_var = eval(self.patch_adatper_name)(
        #     input_var
        # )  # add key "patch tokens" to the dict
        # input_var = eval(self.label_adapter_name)(
        #     input_var
        # )  # add key "label tokens" to the dict
        input_var = self.adapter_rgb(input_var)
        input_var = self.adapter_text(input_var)
        x = self.backbone_module(
            input_var
        )  # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        return x

        x = self.neck_patch(x)
        x = self.neck_label(x)
        decoder_feature = self.decoder_module(x)
        # patch_outputs = self.proj_rgb(decoder_feature)
        label_outputs = self.proj_text(decoder_feature)

        output = {}
        output["pred"] = label_outputs
        # output["pred_patch"] = patch_outputs

        return output
