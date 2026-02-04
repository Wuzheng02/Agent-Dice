
# Agent-Dice

![Agent-Dice Framework](https://github.com/Wuzheng02/Agent-Dice/blob/main/method.png)

## Introduction

**Agent-Dice** is a parameter fusion framework designed to alleviate the plasticity-stability dilemma in the continual learning process of agents. It has been validated on two domains: **tool-use agents** and **GUI agents**, demonstrating its effectiveness in both.

In continual learning, agents often face a trade-off between retaining previously learned knowledge and acquiring new skills. **Agent-Dice** addresses this issue by leveraging **Geometric Consensus** to disentangle knowledge updates, ensuring that agents maintain stability while being able to adapt to new tasks efficiently.

The framework has been shown to significantly improve the performance of agents across a range of tasks, including **tool-use agents** and **GUI-based agents**.

## Key Features

- **Geometric Consensus**: A novel approach for decoupling knowledge updates to preserve both plasticity and stability in agents.
- **Versatile**: Effective for both tool-use agents and GUI agents.
- **Easy-to-use**: Straightforward setup and integration with existing models.
- **Tested**: Comprehensive testing for various domains using `test_loop_gui.py` and `test_loop_tooluse.py`.

## Dataset Preparation

To replicate the results from the paper, you need to prepare the datasets for the two domains.

1. **Tool-Use Agent Data**: Please refer to [tooluse_data.md](https://github.com/Wuzheng02/Agent-Dice/tree/main/Data/tooluse_data.md) for detailed instructions on preparing the dataset.
2. **GUI Agent Data**: Please refer to [gui_data.md](https://github.com/Wuzheng02/Agent-Dice/tree/main/Data/GUI_data.md) for instructions on preparing the dataset.

## Training and Fine-tuning

Once you have prepared the data, you can proceed with fine-tuning using the provided `merge_XXX.py` scripts to fuse parameters with **Agent-Dice**.

### Example Training and Testing:

To fine-tune and test the agent, use the following scripts:

#### For GUI Agent:

```bash
python test_loop_gui.py \
      --model "${model}" \
      --test_json "${test_json}"
````

#### For Tool-Use Agent:

```bash
python test_loop_tooluse.py \
      --model "${model}" \
      --test_json "${test_json}" \
      --data_json "${DATA_JSON}"
```

Where `DATA_JSON` refers to the path of the `data.json` file in the **ToolACE dataset**.

## Dependencies

All training and evaluation is carried out using **llama-factory**. We thank **llama-factory** for their contributions to the community.

* [LlamaFactory Repository](https://github.com/hiyouga/LlamaFactory)



## Citation

If you find our work useful, please cite the following paper:

```bibtex
@article{wu2026agent,
  title={Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning},
  author={Wu, Zheng and Lou, Xingyu and Ma, Xinbei and Li, Yansi and Liu, Weiwen and Zhang, Weinan and Wang, Jun and Zhang, Zhuosheng},
  journal={arXiv preprint arXiv:2601.03641},
  year={2026}
}
```

