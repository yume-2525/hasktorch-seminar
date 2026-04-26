import Torch

trainingData :: [([Int],Int)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

rate :: Tensor
rate = asTensor(0.1 :: Float)

step :: Tensor -> Tensor
step n 
    | n' >= 0.0 = asTensor (1.0 :: Float)
    | otherwise = asTensor (0.0 :: Float)
    where n' = asValue n :: Float

perceptron ::
	Tensor -> -- x
	Tensor -> -- weights
	Tensor -> -- bias
	Tensor    -- output
perceptron x w b = Main.step $ (sumAll $ w * x) + b

caluculateError ::
  Tensor ->
  Tensor ->
  Tensor
caluculateError val ans = ans - val


train_epoch :: 
  [([Int],Int)] -> 
  Tensor -> 
  Tensor -> 
  IO (Tensor, Tensor)
train_epoch [] w b = 
    return (w,b)
train_epoch ((inputlist, ansInt) : rest) w b = do
    let input = asTensor (map fromIntegral inputlist :: [Float])
    let ans = asTensor (fromIntegral ansInt :: Float)
    let error = caluculateError (perceptron input w b) ans

    let neww = w + (rate * error * input)
    let newb = b + rate * error

    train_epoch rest neww newb


train ::
  Int ->  --limit
  [([Int],Int)] -> --input
  Tensor -> --wights
  Tensor -> --bias
  IO (Tensor, Tensor) --output (wights, bias)
train 0 _ w b = do
    putStrLn "Reached the maximum number of iterations."
    return (w, b)

train n input w b = do
    (neww, newb) <- train_epoch input w b
    train (n-1) input neww newb

main :: IO ()
main = do
	-- initialize parameters(weights and bias) with random values.
  x <- randIO [2] defaultOpts
  let w = x * 10
  y <- randIO [] defaultOpts
  let b = y * 10

  let limit = 50000

  (w', b') <- train limit trainingData w b
  putStrLn $ "weight : " ++ show(asValue w' :: [Float]) ++ ", bias : " ++ show(asValue b' :: Float)

  mapM_ (\(inputList, expected) -> do
    let input = asTensor (map fromIntegral inputList :: [Float])
    let target = fromIntegral expected :: Float
    
    let prediction = asValue (perceptron input w' b') :: Float
    
    let rawValue = asValue ((sumAll $ w' * input) + b') :: Float
    
    putStrLn $ "Input: " ++ show(inputList)
            ++ " | Predict: " ++ show(prediction)
            ++ " | Target: " ++(show expected)
            ++ " | Score: " ++ (if prediction == target then "OK" else "NG")
            ++ " (Before step: " ++ show(rawValue)++ ")"
    ) trainingData

  -- train!
  return ()