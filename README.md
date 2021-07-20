# MatrixFactorization for Movielens


## --reference paper
Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30–37. https://doi.org/10.1109/mc.2009.263


## --discription
+ dataset : Movielens
+ predict ratings 0~5


## example : MF
```
python MF.py --path "./datasets/movielens/" --dataset "ratings.csv" --num_factors 20 --epochs 30 --lr 0.01 --regs 0.01 --test_size 0.2
```
