{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude hiding (tanh) 
import Control.Monad (forM_)        --base
--import Data.List (cycle)          --base
--hasktorch
import Torch.Tensor       (asValue,Tensor,asTensor)
import Torch.Functional   (mseLoss, binaryCrossEntropyLoss, Reduction(..))
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
import Data.List
import qualified Data.Vector as V
import qualified Control.Foldl as L
import Text.Read (readMaybe)
import Data.Maybe (fromMaybe)

import Evaluation
import GHC.Generics (Generic)

data Features = Features
  { pclass   :: Float
  , sex      :: Float -- male:0 female:1
  , age      :: Float -- 欠損値は中央値がいいと思ったけど一旦平均値
  , sibSp    :: Float
  , parch    :: Float
  , fare     :: Float
  , embarked :: Float -- C:0 Q:1 S:2 
  } deriving (Show)

data TrainRecord = TrainRecord
  { trainFeatures :: Features
  , survived      :: Float
  } deriving (Show)

data TestRecord = TestRecord
  { testFeatures     :: Features
  , passengerId      :: Float
  } deriving (Show)

data Submission = Submission
  { subPassengerId :: Int
  , subSurvived    :: Int
  } deriving (Show, Generic) 

instance ToNamedRecord Submission
instance DefaultOrdered Submission

sexToFloat :: String -> Float
sexToFloat "male"   = 0.0
sexToFloat "female" = 1.0
sexToFloat _        = 0.0

ageToFloat :: String -> Float
ageToFloat s = case reads s of
                [(val, "")] -> val
                _           -> 29.7

embarkedToFloat :: String -> Float
embarkedToFloat "C" = 0.0
embarkedToFloat "Q" = 1.0
embarkedToFloat "S" = 2.0
embarkedToFloat _   = 2.0

safeRead :: String -> Float
safeRead s = fromMaybe 0.0 (readMaybe s)

normalizeColumn :: [Float] -> [Float]
normalizeColumn xs = 
    let minVal = minimum xs
        maxVal = maximum xs
        range = maxVal - minVal
    in if range == 0 
       then map (const 0.0) xs
       else map (\x -> (x - minVal) / range) xs

loadTrainCSV :: FilePath -> IO [([Float],Float)]
loadTrainCSV path = do
    csvData <- BL.readFile path
    
    case decode NoHeader csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right v -> do
            let records = tail $ V.toList v
                [passengerId, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked] = transpose records

                pclass_f = normalizeColumn $ map safeRead pclass :: [Float]
                sex_f = normalizeColumn $ map sexToFloat sex 
                age_f = normalizeColumn $ map ageToFloat age
                sibsp_f = normalizeColumn $ map safeRead sibsp :: [Float]
                parch_f = normalizeColumn $ map safeRead parch :: [Float]
                fare_f = normalizeColumn $ map safeRead fare :: [Float]
                embarked_f = normalizeColumn $ map embarkedToFloat embarked

                survived_f = map safeRead survived :: [Float]

                features = zipWith7 (\c1 c2 c3 c4 c5 c6 c7 -> [c1, c2, c3, c4, c5, c6, c7])
                           pclass_f sex_f age_f sibsp_f parch_f fare_f embarked_f

            return $ zip features survived_f

loadTestCSV :: FilePath -> IO [([Float],Float)]
loadTestCSV path = do
    csvData <- BL.readFile path
    
    case decode NoHeader csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right v -> do
            let records = tail $ V.toList v
                [passengerId, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked] = transpose records

                pclass_f = normalizeColumn $ map safeRead pclass :: [Float]
                sex_f = normalizeColumn $ map sexToFloat sex 
                age_f = normalizeColumn $ map ageToFloat age
                sibsp_f = normalizeColumn $ map safeRead sibsp :: [Float]
                parch_f = normalizeColumn $ map safeRead parch :: [Float]
                fare_f = normalizeColumn $ map safeRead fare :: [Float]
                embarked_f = normalizeColumn $ map embarkedToFloat embarked

                passengerId_f = map safeRead passengerId :: [Float]

                features = zipWith7 (\c1 c2 c3 c4 c5 c6 c7 -> [c1, c2, c3, c4, c5, c6, c7])
                           pclass_f sex_f age_f sibsp_f parch_f fare_f embarked_f

            return $ zip features passengerId_f
                


device :: Device
device = Device CPU 0

myinputDim :: Int
myinputDim = 7

mylayerSpecs :: [(Int,ActName)]
mylayerSpecs = [(16, Tanh),(1, Sigmoid)]

iter :: Int
iter = 500

learningRate :: Tensor
learningRate = asTensor (2e-2 :: Float)

-- batchSize :: Int
-- batchSize = 10

-- allData :: [([Float],Float)]
-- allData = take 10 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]

border :: Float
border = 0.5

main :: IO()
main = do
  allData <- loadTrainCSV "Session5/titanic/train.csv"
  let n = length allData
  let trainSize = floor (fromIntegral n * 0.8)

  let (trainingData, validData) = splitAt trainSize allData

  let dataCount = asTensor'' device (fromIntegral (length allData) :: Float)
  let tensorData = map (\(input, output) -> (asTensor'' device input, asTensor'' device output)) allData

  let hypParams = MLPHypParams device myinputDim mylayerSpecs
  initModel <- sample hypParams

  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
      let loss = sumTensors $ for tensorData $ \(input,output) ->
                  let y' = mlpLayer model input
                --   in mseLoss output y'
                      w = asTensor'' device (1.0 :: Float)
                  in binaryCrossEntropyLoss ReduceMean output w y'
          meanLoss = loss / dataCount
          lossValue = (asValue meanLoss)::Float 
      showLoss 10 epoc lossValue 
      u <- update model opt meanLoss learningRate
      return (u, lossValue)

  drawLearningCurve "Session5/result_graph/titanic-graph-losses-binaryCrossEntropyLoss-tanh.png" "Learning Curve" [("",reverse losses)]
--   forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do
--       putStr $ show $ input
--       putStr ": "
--       putStrLn $ show ((mlpLayer trainedModel $ asTensor'' device input))
    -- print trainedModel

  let validInputs = map fst validData
  let validActuals = map snd validData

  let predictions = map (\input -> let y' = mlpLayer trainedModel $ asTensor'' device input; y'' = asValue y' :: Float in if y'' >= border then 1 else 0) validInputs

  let actuals = map (\y -> if y >= border then 1 else 0) validActuals

  putStrLn "=== Evaluation Results ==="
  putStrLn $ "Macro F1 : " ++ show (macroF1 actuals predictions)
  putStrLn $ "Weighted  F1 : " ++ show (weightedF1 actuals predictions)
  putStrLn $ "Micro F1 : " ++ show (microF1 actuals predictions)

  testDataRaw <- loadTestCSV "Session5/titanic/test.csv"
  
  let submissionData = map (\(features, pId) -> 
          let y' = mlpLayer trainedModel $ asTensor'' device features
              y'' = asValue y' :: Float
              predicted = if y'' >= border then 1 else 0
          in Submission (round pId) predicted
        ) testDataRaw

  let csvContent = encodeDefaultOrderedByName submissionData
  BL.writeFile "Session5/titanic/submission.csv" csvContent


  where for = flip map