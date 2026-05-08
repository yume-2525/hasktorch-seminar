## 1. Build and train an AND gate using a simple perceptron
#### Process

Based on template, the following points can be considered.

step  
* argument１：Tensor（Scalar）、sum（input）＊（weight）
* return value：Tensor（Scalar）、result
* behavior：A function that takes a scalar tensor as input and returns 1 if it is positive, and returns 0 otherwise. 

percepron  
* argument１：Tensor（2vector）、input
* argument２：Tensor（2vector）、weight
* argument３：Tensor（Scalar）、bias
* return value：Tenor（Scalar）、input to the activation function
* behavior：A function that takes (x, y) as input and returns what it computes the sum including the weights and bias.

caluclateError  
* argument１：Tensor（Scalar）、stepのresult
* return value：Tensor（Scalar）、error
* behavior：A function that cluclates error.
* notes：Dose it prefer that I chenge two argments and make a function that returns the result by seeing result and correct value?

Also、I read [Note](https://medium.com/analytics-vidhya/implementing-perceptron-learning-algorithm-to-solve-and-in-python-903516300b2f) and understood the following flow.

1. Set random wight, bias, and training sets. Culculate （Σ（input）＊（weight））＋（bias）
2. Pass the value through the step function.
3. Perform the above process in the perceptron function to obtain the predicted value.
4. Compute the error using calculateError.
5. Set the maximum number of iterations and the learning rate. Repeat updating the weights and bias until the sum of errors becomes zero or the numper of iterations reaches the maximum limit. The wights are updated using (new value) = (old value) + (leaning rate) * (error) * (corresponding input), and the bias is updated using (new value) = (old value) + (leaning rate) * (error).
![](./memo_02.png)

The result can be predicted as follows.
![](./memo_01.png)

Question：When are the weights updated?  
→The weights are updated after each data sample, not after each epoch.

#### executing result
```
weight : [4.42507,4.7318616], bias : -4.811824
Input: [1,1] | Predict: 1.0 | Target: 1 | Score: OK (Before step: 4.345107)
Input: [1,0] | Predict: 0.0 | Target: 0 | Score: OK (Before step: -0.38675404)
Input: [0,1] | Predict: 0.0 | Target: 0 | Score: OK (Before step: -7.996225e-2)
Input: [0,0] | Predict: 0.0 | Target: 0 | Score: OK (Before step: -4.811824)
```

------------


## 2. Build a XOR gate using a multi-layer perceptron and train it employing the backpropagation mechanism available in the hasktorch library.

### b. Familiarize yourself with the hasktorch implementation, ensuring its alignment with the video's explanation. 
```
data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }
  ```
  Structure of multilayer perceptron  
  feature_counts：the number of parameters in each layer　
  nonlinearitySpec：the activation function

  ```
  data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)
  ```
> data Linear  
> weight :: Parameter	   
> bias :: Parameter  
  Contents of the multilayer perceptron  
  layers：weights and biases of each layer  
  nonlinearity：activation function

```
instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)
```
> class Randomizable spec f | spec -> f where  
> sample :: spec -> IO f  
> sample：function　spec：structure　f：contents　（Create value containing random numbers)

>scanl :: (b -> a -> b) -> b -> [a] -> [b]   
>scanl is similar to foldl, but returns a list of successive reduced values from the left:  
>scanl f z [x1, x2, ...] == [z, z `f` x1, (z `f` x1) `f` x2, ...]

>uncurry :: (a -> b -> c) -> (a, b) -> c  
>uncurry converts a curried function to a function on pairs.  
>e.g.  uncurry (+) (1,2) >> 3, map (uncurry max) [(1,2), (3,4), (6,8)] >> [2,4,8]

> data LinearSpec   
> in_features :: Int     
> out_features :: Int  
> in_features：input size　out_features：出力 size

Behavior：Teceives an MLPSpec and returns an MLP filled with random values of the specified shape.
mkLayerSizes：A function that takes a list and returns a list of adjacent pairs. e.g. [1,2,3]->[(1,2),(2,3)]  
layer_sizes：a list of （input size、output size） e.g. [（number of parameters in the １ layer、number of parameters in the ２ layer）,（number of parameters in the 2 layer、number of parameters in the 3 layer）..]  
linears：layer_sizes conberted into the LinearSpec type 
sample：a function that takes a record with the structure of NLPSpec and returns an MLP


```
mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x
```
> intersperse :: a -> [a] -> [a]  
> The intersperse function takes an element and a list and `intersperses' that element between the elements of the list.  
> e.g. intersperse ',' "abcde" >> "a,b,c,d,e"、intersperse 1 [3, 4, 5] >> [3,1,4,1,5]  

> linear :: Linear -> Tensor -> Tensor  
（A funbrion that computes（input値）＊（weight）＋（bias）for one layer）

argument１：MLP、the contents of the multilayer perceptron  
argument２：Tensor（vector）、input values  
return value：Tensor（vector）、result
behavior：Computes the result using the wights, biases, and activation functions of each layer for the given input values, and returns the resulting vector.


```
batchSize = 2
```
The number of data sumples used in one loop.

```
numIters = 2000
```
The maxmum number of iterations

```
model :: MLP -> Tensor -> Tensor
model params t = mlp params t
```
argument１：MLP、Contents of multilayer perceptron
argument２：Tensor（vector）、input value
return value：Tensor（vector）、result
behavior：Computes the result using the wights, biases, and activation functions of each layer for the given input values, and returns the resulting vector.（A function uses the mlp）

```
main :: IO ()
main = do
  init <-
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1],
          nonlinearitySpec = Torch.tanh
        }
```
In init, a multilayer perceptron is initialized with 2 input nodes, 2 hidden-layer nodes, 1 output node, the tanh activation functionm and randomly initialized weights.

```
  trained <- foldLoop init numIters $ \state i -> do
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
```
> ">>="：Sequentially compose two actions, passing any value produced by the first as an argument to the second. 'as >>= bs' can be understood as the do expression  
> return . (toDType Float) . (gt 0.5)：. is the function composition operatorm and the functions are applied from right to left. Therefore, it first checks whether the value is freater than 0.5, then converts the result to Float, and finally returns it.

trained stores the repeatedly trained weights and other parameters(state).
input is randomly generated training data.
```
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
```
> squeezeAll :: Tensor -> Tensor  
> Remoces all unnecessary dimensions whose size is 1. Since the output size is 1 in this case, the Tensor is compressed into a one-dimensional vector. 

Calclates the error.  

```
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newState, _) <- runStep state optimizer loss 1e-1
    return newState
```
> runStep :: (Parameterized model, Optimizer optimizer) => 
> model -> optimizer -> Tensor -> LearningRate -> IO (model, optimizer)  
> argument１：model、the data currently being trained 
> argument２：optimizer、the update method  
> argument３：Tensor(Scalar)、error  
> argument４：LearningRate 、learning rate  
> return value：（updated data、update history）  

Updates state using gradient descent with a learning rate of 1e-1  

```
  putStrLn "Final Model:"
  putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 0 :: Float]))
  putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 1 :: Float]))
  putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 0 :: Float]))
  putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 1 :: Float]))
  return ()
  where
    optimizer = GD
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
```
Test using final value.

------------------

### c. Analyze the differences between this implementation and the hasktorch version. Modify your code for enhanced readability.

####Code Analysis

```
trainingData :: [([Float],Float)]
trainingData = take 10 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]
```
Creates 10 training data samples.  

```
main :: IO()
main = do
  let iter = 1500::Int
      device = Device CUDA 0
      hypParams = MLPHypParams device 2 [(3,Sigmoid),(1,Sigmoid)]
```
> data MLPHypParams = MLPHypParams {  
> dev :: Device,  
> inputDim :: Int,  
> layerSpecs :: [(Int,ActName)]  
> } deriving (Eq, Show)  
Constructs the configuration of the perceptron. In this case, the perceptron uses Device CUDA 0, has 2 nodes in the input layer, 3 nodes in the hiddin layer, and 1 node in the outout layer. The sigmoid function is used as the activation function between the input and hidden layers, as well as between the hidden and output layers.

```
  initModel <- sample hypParams
```
initModel is assigned a structure based on hyParames and filled with random values.  

```
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let loss = sumTensors $ for trainingData $ \(input,output) ->
                  let y = asTensor'' device output
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'
        lossValue = (asValue loss)::Float 
    showLoss 10 epoc lossValue 
    u <- update model opt loss 1e-1
    return (u, lossValue)
```
> mapAccumM :: (Monad m, Foldable t) => t a -> b -> (a -> b -> m (b,c)) -> m (b, [c])  
> mapAccumM xs zero f = do  
>  foldM (\(prev,lst) x -> do  
>                           bc <- f x prev  
>                           return (fst bc, (snd bc):lst)  
>                           ) (zero,[]) xs  
> behavior：A function that updates the state while also keeping a record of each stap.  
> argument１：foldable a, a list of iteration counts  
> argument２：b、the initial value  
> argument３：(a->b->m (b,c))、function  

> for：a function with the arguments of map reversed  

> asTensor''：a function that creates Tensor data while specifying the device on which it is stored  

> mlpLayer：a function that calculates the output value given a perceptron model and input values  

> mseLoss：a function that calculates the mean squared error  

> sumTensors：sums the mean squared errors of the 10 traing data samples  

> showLess：displays the error once every 10 iterations  

> update：a function that takes the current model, optimization method, error, and learning rate, and returns the improved model  

Behavior：trainModel and losses are assigned the perceptron model obtained by updation the initial value initModel using GD(gradient descent) for iter iterations, along with the errors during training.


```
  drawLearningCurve "graph-xor.png" "Learning Curve" [("",reverse losses)]
```
> drawLearningCurve  
> argument１：ファイル名
> argument２：グラフのタイトル
> argument３：[（name、a list of data）]
Plot the learning curve.  
![](./graph-xor.png)

```
  forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do
    putStr $ show $ input
    putStr ": "
    putStrLn $ show ((mlpLayer trainedModel $ asTensor'' device input))
  -- print trainedModel
  where for = flip map
  ```
Display the final training result.  
```
[1.0,1.0]: Tensor Float []  5.4089e-2
[1.0,0.0]: Tensor Float []  0.9395   
[0.0,1.0]: Tensor Float []  0.9266   
[0.0,0.0]: Tensor Float []  7.2013e-2
```

####Differences between the two codes
+ Uses the hasktorch-tools set（MLPHypParams, mlpLayer, updateなど）, making the code more concise.
+ Difines the device used to store the data
+ Records the update process during training.
+ Defines numerical values othe than the training data indside the main function.

####Modifications to improve readaility
Variables were defined outside the main functio a s shown below, making the main function simpler and easier to read.  
```
device :: Device
device = Device CPU 0

myinputDim :: Int
myinputDim = 2

mylayerSpecs :: [(Int,ActName)]
mylayerSpecs = [(3,Sigmoid),(1,Sigmoid)]

iter :: Int
iter = 1500

learningRate :: Tensor
learningRate = asTensor (1e-1 :: Float)

batchSize :: Int
batchSize = 10

trainingData :: [([Float],Float)]
trainingData = take 10 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]
```

------------------

### b. Experiment with XOR using a step function.
Referring to MlpXor.hs, the activation function was changed to a step function and the code was executed.   
→An error occurred and no result was produced. 

**Discussion**  
The step function is not differentiable, and its gradient is either 0 or changes abruptly. As a result, an appropriate gredient cannnot be computed, making it impossible to properly update the model.
