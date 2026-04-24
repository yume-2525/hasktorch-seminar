module LinearRegression where
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, add, transpose2D, sumAll)
import ML.Exp.Chart (drawLearningCurve)


ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

sampleA :: Tensor
sampleA = asTensor (0.555 :: Float)
sampleB :: Tensor
sampleB = asTensor (94.585026 :: Float)

h :: Tensor
h = asTensor (0.0001 :: Float)
rate :: Tensor
rate = asTensor (0.0000002 :: Float)

linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = slope * input + intercept

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = sumAll $ (z - z')^2

calculateNewA :: 
     Tensor ->
	 Tensor ->
     Tensor
calculateNewA a b = (cost (linear (a+h, b) xs) ys - cost (linear (a, b) xs) ys) / h

calculateNewB :: 
     Tensor ->
	 Tensor ->
     Tensor
calculateNewB a b = (cost (linear (a, b+h) xs) ys - cost (linear (a, b) xs) ys) / h

train :: Int -> Tensor -> Tensor -> [Float] -> IO(Tensor, Tensor, [Float])
train 0 a b list= do
	let cost_c = cost ys ((linear (a, b)) xs)
	putStrLn $ show(0) ++ " : cost->" ++ show(asValue cost_c :: Float) ++ " a->"  ++ show(asValue a :: Float) ++ " b->"  ++ show(asValue b :: Float)
	putStrLn "学習終了"
	let newlist = list ++ [(asValue cost_c :: Float)]
	return (a, b, newlist)

train n a b list = do
	let cost_c = cost ys ((linear (a, b)) xs)
	putStrLn $ show(n) ++ " : cost->" ++ show(asValue cost_c :: Float) ++ " a->"  ++ show(asValue a :: Float) ++ " b->"  ++ show(asValue b :: Float)

	let newa = a - (rate * (calculateNewA a b))
	let newb = b - (rate * (calculateNewB a b))
	let newlist = list ++ [(asValue cost_c :: Float)]
	train (n-1) newa newb newlist 


main :: IO ()
main = do
   
  (a,b,d) <- train 30 sampleA sampleB []
  drawLearningCurve "train_result2.png" "learning curve" [("3.g", d)]

  -- Below are pseudo code

  -- Iterate through the provided xs and ys data. 
  -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.
  
  -- let estimatedY = linear (sampleA, sampleB) xs 
  -- let ysList = asValue ys :: [Float]
  -- let estimatedYList = asValue estimatedY :: [Float]
  -- let result = zip ysList estimatedYList

  -- mapM_ (\(y, estimatedY) -> do
  --   print $ "correct answer: " ++ show y
  --   print $ "estimated: " ++ show estimatedY
  --   print $ "******"
  --   ) result
  -- for x, y in (xs, xy)
    -- convertToTensor x
	--   estimatedY = linear (sampleA, sampleB) x
	--   print "correct answer:" + y
	--   print "estimated: " + estimatedY
	--   print "******"
	 
	-- Expected outputs:
	-- correct answer: 148
	-- estimated: ?
	-- *******
	-- correct answer: 186
	-- ...
  return ()