{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Admit_kld where

import Prelude hiding (tanh) 
import Control.Monad (forM_)        --base
--import Data.List (cycle)          --base
--hasktorch
import Torch.Tensor       (asValue,Tensor,asTensor)
import Torch.Functional   (mseLoss, binaryCrossEntropyLoss, nllLoss', klDiv, logSoftmax,Dim(..), Reduction(..))
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (sample)
import Torch.Train        (update,showLoss,sumTensors)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tool

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import qualified Control.Foldl as L

import Evaluation

loadCSV :: FilePath -> IO [([Float],Float)]
loadCSV path = do
    csvData <- BL.readFile path
    
    case decode NoHeader csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right v -> do
            let records = V.toList v
                (ys, x1s, x2s) = unzip3 records

                meanx1 = L.fold L.mean x1s
                stdx1 = L.fold L.std x1s
                meanx2 = L.fold L.mean x2s
                stdx2 = L.fold L.std x2s
                myList  = map (\(y, x1, x2) -> ([(x1-meanx1)/stdx1, (x2-meanx2)/stdx2],y)) records
            
            return myList

device :: Device
device = Device CPU 0

myinputDim :: Int
myinputDim = 2

mylayerSpecs :: [(Int,ActName)]
mylayerSpecs = [(5, Tanh),(2, Id)]

iter :: Int
iter = 1300

learningRate :: Tensor
learningRate = asTensor (1e-2 :: Float)

-- batchSize :: Int
-- batchSize = 10

-- trainingData :: [([Float],Float)]
-- trainingData = take 10 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]

border :: Float
border = 0.75

main :: IO()
main = do
  trainingData <- loadCSV "Session5/data/train.csv"
  testData <- loadCSV "Session5/data/eval.csv"

  let dataCount = asTensor'' device (fromIntegral (length trainingData) :: Float)
--   let tensorData = map (\(input, output) -> (asTensor'' device input, asTensor'' device output)) trainingData
  let tensorData = map (\(input, output) -> 
            let targetDist = if output >= border then [0.0, 1.0] else [1.0, 0.0] :: [Float]
            in (asTensor'' device input, asTensor'' device targetDist)
          ) trainingData
  let hypParams = MLPHypParams device myinputDim mylayerSpecs
  initModel <- sample hypParams

  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
      let loss = sumTensors $ for tensorData $ \(input,output) ->
                --   let y' = mlpLayer model input
                --   in mseLoss output y'
                    --   w = asTensor'' device (1.0 :: Float)
                --   in binaryCrossEntropyLoss ReduceMean output w y'
                  let rawOutput = mlpLayer model input
                      logProbs = logSoftmax (Dim 0) rawOutput 
                  in klDiv ReduceMean logProbs output
          meanLoss = loss / dataCount
          lossValue = (asValue meanLoss)::Float 
      showLoss 10 epoc lossValue 
      u <- update model opt meanLoss learningRate
      return (u, lossValue)

  drawLearningCurve "Session5/result_graph/graph-losses-KLDLoss-tanh.png" "Learning Curve" [("",reverse losses)]
--   forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do
--       putStr $ show $ input
--       putStr ": "
--       putStrLn $ show ((mlpLayer trainedModel $ asTensor'' device input))
    -- print trainedModel

  -- (これより上の学習ループ部分はそのまま)

  let testInputs = map fst testData
  let testActuals = map snd testData

  let predictions = map (\input -> let y' = mlpLayer trainedModel $ asTensor'' device input; y'' = asValue y' :: Float in if y'' >= border then 1 else 0) testInputs

  let actuals = map (\y -> if y >= border then 1 else 0) testActuals

  putStrLn "=== Evaluation Results ==="
  putStrLn $ "Macro F1 : " ++ show (macroF1 actuals predictions)
  putStrLn $ "Weighted  F1 : " ++ show (weightedF1 actuals predictions)
  putStrLn $ "Micro F1 : " ++ show (microF1 actuals predictions)

  where for = flip map