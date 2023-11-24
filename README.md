# CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation

Official implementation of CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation (NeurIPS 2023).

## Abstract

Label noise widely exists in large-scale image datasets. To mitigate the side effects of label noise, state-of-the-art methods focus on  selecting confident examples by leveraging semi-supervised learning.  Existing research shows that the ability to extract hard confident  examples, which are close to the decision boundary, significantly  influences the generalization ability of the learned classifier. In this paper, we find that a key reason for some hard examples being  close to the decision boundary is due to the entanglement of style  factors with content factors. The hard examples become more  discriminative when we focus solely on content factors, such as semantic information, while ignoring style factors. Nonetheless, given only  noisy data, content factors are not directly observed and have to be  inferred. To tackle the problem of inferring content factors for classification  when learning with noisy labels, our objective is to ensure that the  content factors of all examples in the same underlying clean class  remain unchanged as their style information changes. To achieve this, we utilize different data augmentation techniques to  alter the styles while regularizing content factors based on some  confident examples. By training existing methods with our inferred  content factors, CS-Isolate proves their effectiveness in learning hard  examples on benchmark datasets.

## Experiments

To install the necessary Python packages:

```
pip install -r requirements.txt
```

Download the CIFAR-10 and CIFAR-100 datasets:

```
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
mv cifar-10-batches-py cifar-10
wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
mv cifar-100-batches-py cifar-100
```

For the FashionMNIST dataset, we use the dataset provided by [PTD](https://github.com/xiaoboxia/Part-dependent-label-noise). The images and labels have been processed to .npy format. You can download the [fashionmnist](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) here.

To train the model on FashionMNIST:

```
python main.py --noise_mode instance --dataset fashionmnist --data_path ./fashionmnist --num_class 10 --r 0.4
```

To train the model on CIFAR-10:

```
python main.py --noise_mode instance --dataset cifar10 --data_path ./cifar-10 --num_class 10 --r 0.4
```

To train the model on CIFAR-100:

```
python main.py --noise_mode instance --dataset cifar100 --data_path ./cifar-100 --num_class 100 --r 0.4 --lambda_u 100
```

To train the model on CIFAR-10N:

```
python main.py --noise_mode worse_label --dataset cifar10 --data_path ./cifar-10 --num_class 10
```

## Citation

If you find our work insightful, please consider citing our paper:

```
@inproceedings{lin2023cs,
  title={CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation},
  author={Lin, Yexiong and Yao, Yu and Shi, Xiaolong and Gong, Mingming and Shen, Xu and Xu, Dong and Liu, Tongliang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

