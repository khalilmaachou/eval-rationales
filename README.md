# Eval-Rationales

Eval-Rationales is an end-to-end toolkit to Explain and Evaluate Transformers-Based models.

This library unifies the explainabilites techniques and evaluations applied to Interpertable Models. The code in this library is mainly based on:
- 🤗 [HuggingFace models](https://huggingface.co/models)
- 🤗 [Datasets](https://huggingface.co/datasets)
- [Captum](https://captum.ai/)
- [ERASER benchmark](https://github.com/successar/Eraser-Benchmark-Baseline-Models/tree/master)

## Installation

Create a new pip environment for this library `eval-expl` and install the dependent libraries.

```bash
mkdir eval-expl
virtualenv --system-site-packages eval-expl/
source eval-expl/bin/activate
pip install -r requirements.txt --user
export PYTHONPATH=.
```
## Usage:

Eval-Rationales is mainly executed on the `predict.py` script and it considers two type of parameters:


### Required: 
- `--data`: the path to a Datasets object. It can be a Datasets URL or local path storage (for example for the MIMIC dataset, check also optional parameters )
- `--model`: the path to the model object. It is based on the HuggingFaces library. Similar to data, this path can be a URL for the HuggingFaces Hub, or a local path (for example if you want to use the fine-tuned modesl from the ERASER benchmark). It can also handle DecisionTree or LogisticRegression models (loaded from a pickle file).
- `--saliency`: it refers to the XAI technique. Currently, there are four available methods: *lime*, *random*, *gradient*, and *attention*.

### Optional
- `--split`: you can specifiy the split of the dataset you want to use (train, test, validation).

- `--metrics`: to enable the metrics calculation set it to True. Otherwise the library will not do evaluation phase (it will only create instance of model, dataset, and the explainer).
- `--visualize`: set it to True if you want to visualize the tokens scores in the input.

In case you want to use MIMIC dataset these two parameters need to be added in the command line
- `--task`: references to the MIMIC task
- `--mimic_path`:  references to the absolute path to the mimic data folder.

If you want to use ERASER pre-trained models you need to add the predictor_type
- `--predictor_type`: see [ERASER repository](https://github.com/successar/Eraser-Benchmark-Baseline-Models/tree/master) for more information on predictor_types.

  
  


## Example
The following command line use the available `bert-base-uncased` from the HuggingFace hub and the `movie_rationales` dataset from the Datasets Hub. It computes metrics using the XAI attention technique.

```bash
    python predict.py --data movie_rationales --model bert-base-uncased --saliency attention
```


## Considerations
- To handle a Datasets object, we assume the dataset contains at least two attributes, first is the input (text column) and the second is the label.
- If you want to use the ERASER models, please do not change the format and keep them as an .tar.gz files.
- When you want to do the explainability using ERASER code, keep in mind that the batch_size used in the train phase and prediction phase need to be the same.

## Citing
For further details, please refer to the paper:
```tex
@inproceedings{maachou2024eval,
  title={eval-rationales: An End-to-End Toolkit to Explain and Evaluate Transformers-Based Models},
  author={Maachou, Khalil and Lov{\'o}n-Melgarejo, Jes{\'u}s and Moreno, Jose G and Tamine, Lynda},
  booktitle={European Conference on Information Retrieval},
  pages={212--217},
  year={2024},
  organization={Springer}
}
```


## Contact
If you have any problem at all or any suggestions that can optimize this library, you can always contact me using my personal mail: khalilmaachou.99@gmail.com or using this discord server: https://discord.gg/eDTC6688WX.

Tutorial using this library [HERE](https://youtu.be/3M1MJPhmMQE).
