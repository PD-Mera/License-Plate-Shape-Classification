# License Plate Shape Classification

This project simply classifies whether license plate shape is **Square** or **Rectangle**

## Environments

- Python 3.10.6

Install requirements

``` bash
pip install -r requirements.txt
```

For more infomation, refer to [DEVICE.md](./DEVICE.md)

## Data

- I cannot public my data. You can use your own data or crawl from internet.

- Data in this format

``` files
|-- data
    |-- train
    |   |-- class 1
    |   |-- class 2
    |   `-- ...
    |-- valid
    |   |-- class 1
    |   |-- class 2
    |   `-- ...
    `-- test
        |-- class 1
        |-- class 2
        `-- ...
```

### Config

Modify infomation about training in `config.py`

### Train

Simply run 

``` bash
python train.py
```

### Experiment Results

Some experiment results

| Model             | Training Info | Epoch | Best Accuracy | Pretrained                      |
| ----------------- |:-------------:| :---: | :-----------: | :-----------------------------: |
| Resnet18          | Adam, lr=1e-5 | 10    | 99.89%        | [Model](https://bit.ly/3PshMuc) |
| MobilenetV3_Small | Adam, lr=1e-5 | 30    | 100.00%       | [Model](https://bit.ly/3PNqKST) |

You can download weight file above and put in `weights` folder and run inference

``` bash
python infer.py
```

#### Some inference results

- Rectangle Image

![Rectangle](assets/rectangle.jpg "Rectangle Image")

- Square Image

![Square](assets/square.jpg "Square Image")