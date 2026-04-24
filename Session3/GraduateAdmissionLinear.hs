module Main where
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, add, transpose2D, sumAll)
import ML.Exp.Chart (drawLearningCurve)

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import Torch.Tensor (Tensor, asTensor)

loadCSV :: FilePath -> IO (Tensor, Tensor, Tensor)
loadCSV path = do
    csvData <- BL.readFile path
    
    case decode NoHeader csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right v -> do
            let records = V.toList v
                
                ysList  = map (\(y, _, _) -> y) records
                x1List  = map (\(_, x1, _) -> x1) records
                x2List  = map (\(_, _, x2) -> x2) records
            
            return (asTensor (ysList :: [Float]), asTensor (x1List :: [Float]), asTensor (x2List :: [Float]))



sampleA :: Tensor
sampleA = asTensor (0 :: Float)
sampleB :: Tensor
sampleB = asTensor (0 :: Float)
sampleC :: Tensor
sampleC = asTensor (0 :: Float)

h :: Tensor
h = asTensor (0.0001 :: Float)
rate :: Tensor
rate = asTensor (0.0000002 :: Float)

linear :: 
    (Tensor, Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor -> 
    Tensor              -- ^ z: 1 × 10
linear (slope1, slope2, intercept) input1 input2 = slope1 * input1 + slope2 * input2 + intercept

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = sumAll $ (z - z')^2

calculateNewA :: 
     Tensor ->
	 Tensor ->
     Tensor ->
     Tensor ->
	 Tensor ->
     Tensor ->
     Tensor
calculateNewA a b c x1 x2 ys = (cost ys (linear (a+h, b, c) x1 x2) - cost ys (linear (a, b, c) x1 x2)) / h

calculateNewB :: 
     Tensor ->
	 Tensor ->
     Tensor ->
     Tensor ->
	 Tensor ->
     Tensor ->
     Tensor
calculateNewB a b c x1 x2 ys = (cost ys (linear (a, b+h, c) x1 x2) - cost ys (linear (a, b, c) x1 x2)) / h

calculateNewC :: 
     Tensor ->
	 Tensor ->
     Tensor ->
     Tensor ->
	 Tensor ->
     Tensor ->
     Tensor
calculateNewC a b c x1 x2 ys = (cost ys (linear (a, b, c+h) x1 x2) - cost ys (linear (a, b, c) x1 x2)) / h


train :: Int -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> [Float] -> IO(Tensor, Tensor, Tensor, [Float])
train 0 a b c x1 x2 ys list = do
    let cost_c = cost ys ((linear (a, b, c)) x1 x2)
    putStrLn $ "探索終了->" ++ show(asValue cost_c :: Float)
    return (a, b, c, list ++ [asValue cost_c :: Float])

train n a b c x1 x2 ys list = do
    let cost_c = cost ys ((linear (a, b, c)) x1 x2)
   
    putStrLn (show n ++ " : cost->" ++ show(asValue cost_c :: Float))

    let newa = a - (rate * calculateNewA a b c x1 x2 ys)
    let newb = b - (rate * calculateNewB a b c x1 x2 ys)
    let newc = c - (rate * calculateNewC a b c x1 x2 ys)
    
    train (n-1) newa newb newc x1 x2 ys (list ++ [asValue cost_c :: Float])


main :: IO ()
main = do
   
  (ys, x1, x2)  <- loadCSV "Session3/data/train.csv"

  (a,b,c, d) <- train 100 sampleA sampleB sampleC x1 (x2/100.0) ys []
  drawLearningCurve "5_train.png" "learning curve" [("5_train", d)]

  putStrLn $ " : a->" ++ show(asValue a :: Float) ++" : b->" ++ show(asValue b :: Float) ++ " : c->" ++ show(asValue c :: Float)

  return ()