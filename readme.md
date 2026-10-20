# DiffGOP: An End-to-End Diffusion Model for Gaze Object Prediction
Official repository for DiffGOP: An End-to-End Diffusion Model for Gaze Object Prediction, accepted by .

We provides the implementation of **DiffGOP**, an end-to-end diffusion framework for gaze object prediction. DiffGOP employs frequency-aware human-centric conditioning to capture human–target interactions and guide iterative denoising. It includes two approaches: **DiffGOP-BD**, which progressively refines gaze-target bounding boxes through box-level diffusion, and **DiffGOP-QD**, which refines gaze-target predictions through query-level diffusion. The repository provides:

   - code and pretrained model weights of DiffGOP-BD and DiffGOP-QD.
   - Data of GOO_Synth, GOO_Real and DiSG.

![Introduction of DiffGOP](assets/introduction.png)

<p align="center">
  <em>Figure 1. Comparison between conventional one-step prediction and DiffGOP, which progressively refines gaze-target predictions through iterative diffusion denoising.</em>
</p>


# DiffGOP Framework
![DiffGOP Framework](assets/Framework.png)

<p align="center">
  <em>Overall architecture of DiffGOP.</em>
</p>

![Different Objects](assets/differentOB.png)


<p align="center">
  <em>Box-level denoising in DiffGOP-BD and query-level denoising in DiffGOP-QD.</em>
</p>


## DiffGOP-BD Environment Setup

```bash
cd DiffGOP-BD
conda env create -f environment.yaml
conda activate DiffGOP-BD
The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2).
```

