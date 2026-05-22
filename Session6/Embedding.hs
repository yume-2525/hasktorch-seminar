{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

-- module Main (main) where
module Embedding (main) where

import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import qualified Data.ByteString.Lazy.Char8 as BC
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List (nub)
import Data.Char (toLower)

import Torch.DType (DType(..))
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', logSoftmax, nllLoss', sumDim, Dim(..), KeepDim(..),unsqueeze)
import Torch.NN (Parameterized(..), Parameter,sample)
import Torch.Train        (update,showLoss,sumTensors)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (eye', zeros')
import Torch.Layer.MLP    (MLPParams, MLPHypParams(..),ActName(..),mlpLayer)
import Torch.Device       (Device(..),DeviceType(..))
import ML.Exp.Chart   (drawLearningCurve) --nlp-tool

-- your text data (try small data first)
textFilePath = "Session6/data/sample3.txt"                -- 訓練に使うテキスト
modelPath =  "Session6/data/sample_embedding4-3.params"     -- 作成したMLP
wordLstPath = "Session6/data/sample_wordlst4-3.txt"         -- 作成した単語リスト
validFilePath = "Session6/data/valid.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words　総数
  wordDim :: Int  -- the dimention of word embeddings　ベクトルの次元数
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter  -- 重み本体 Paramaterは重みを更新できる型
  } deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model {
		mlp :: MLPParams,             -- MLPの構成
    embeddings :: Embedding     -- MLPに突っ込む重み
  } deriving (Generic, Parameterized)

-- 記号を判定する
isUnncessaryChar :: 
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", ",", "!", "?", "<br", "<", "/", ">", "(", ")", "*", "@", "-", "&", "#", ";", ":", "\\", "[", "]", "_"]
--リストに含まれた文字について、encode([String]を[Word8型]へ)したhead(前から順に。すなわち[Word8]->Word8)を持ってきて、strとelem(要素であるかどうか判断し)、Bool型を出力。

-- テキストを記号を排除した単語ごとのリスト[[B.ByteString]](行数*単語数)にする
preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
-- 単語ごとに区切る　 ["I am a dog.", "I don't like dogs"]->[["I", "am", "a", "dog"], ["I", "don't", "like", "dogs"]]
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    -- 記号を排除する "Hi!" -> "Hi"
    -- unpack(ByteString入力値->[Word8])で文字列を数字のリストに変換, isUnncessaryCharの結果をnot(Bool型を反転)し、filter（Trueになる要素だけ抜き出す）にかけて、再度pack([Word8]->ByteString)する。
    textLines = B.split (head $ encode "\n") filteredtexts
    -- 行ごとに区切る　"I am a dog.\nI don't like dogs"→["I am a dog.", "I don't like dogs"]
    -- encodeで"\n"を[Word8]型に直しheadでWord8にする。splitで"\n"ごとに区切る

-- 単語リストを使って、単語を渡すと単語のIDを教えてくれる関数を作る
wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))
-- (zip wordlst [0.. length wordlst])：["This", "is", "awesome"] -> [("This", 0), ("is", 1), ("awesome"", 2)]
-- M.findWithDefault len wrd list : listにwrdがあればその番号を返し、なければlenを返す

-- MLPの初期値を作成
toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim
-- eye' a b : a*bの単位行列を作る

cbow :: Model -> Tensor -> Tensor
cbow model inputTensor = 
  let weightTensor = toDependent $ wordEmbedding (embeddings model)   -- modelの中のembeddingsを取り出す
      embVecs = embedding' weightTensor inputTensor                   -- 入力された位置の重みを二つ分とってくる
      margedVec = sumDim (Dim 0) RemoveDim Float embVecs                    -- 二つの重みを足し合わせて一つのベクトルを作る
  in mlpLayer (mlp model) margedVec                                   --　 mlpに入力値として渡す


iter :: Int
iter = 1500

learningRate :: Tensor
learningRate = asTensor (0.5 :: Float)

batchsize :: Int
batchsize = 128


main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath
  let lowerTexts = BC.map toLower texts -- 小文字に直す

  -- Create a unique word list
  let wordLines = preprocess lowerTexts              -- [["Hi"],["Good","day"],["Nice","day"]]
      wordlst = nub $ concat wordLines          -- ["Hi", "Good", "day", "Nice"]
      -- concat [[a]] -> [a] : 二次元リストを一次元にならす
      -- nub [a] -> [a] : ユニークなものだけ残す
      wordToIndex = wordToIndexFactory wordlst  -- [("Hi",0), ("Good",1), ("day",2), ("Nice",3)]
  print wordlst

  let mywordDim = 50
  let mylayerSpecs = [((length wordlst) + 1, Id)]
  let hypParams = MLPHypParams (Device CPU 0) mywordDim mylayerSpecs
  initMlp <- sample hypParams

  -- Create initial embedding (wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst + 1, wordDim = mywordDim}
  -- MLPの型を作る
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  -- makeIndependent : Tensorを更新できる重み（Parameter）に変換する関数
  let emb = Embedding { wordEmbedding = wordEmb }
  -- 初期値の入ったEmbeddingを作る

  let initModel = Model {
		mlp = initMlp,         
    embeddings = emb
  }

  -- let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
  -- -- convert word to index
  --     idxes = map (map wordToIndex) (preprocess sampleTxt)
  -- -- convert to embedding
  --     embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)
  -- -- embedding' :: Tensor(weights) -> Tensor(indices) -> Tensor：指定した単語の重み(ベクトル)をとってくる


  -- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.
  -- train用とvalid用のデータをMLPが訓練できる形([前の単語のインデックス,後の単語のインデックス],正解のインデックス)のリストに切り分ける。
  let wordlstAll = concat wordLines -- １行にまとめる
  let wordindex = map wordToIndex wordlstAll
  let wordlstSplit = zip3 wordindex (tail wordindex) (tail $ tail wordindex)
  let dataset = map (\(prev, curr, next) -> ([prev, next], curr)) wordlstSplit
  let dataCount = asTensor ((fromIntegral (length wordlstAll)) - 2 :: Float)

  let validlstAll = concat wordLines -- １行にまとめる
  let validindex = map wordToIndex validlstAll
  let validlstSplit = zip3 validindex (tail validindex) (tail $ tail validindex)
  let validset = map (\(prev, curr, next) -> ([prev, next], curr)) wordlstSplit
  let validCount = asTensor ((fromIntegral (length validlstAll)) - 2 :: Float)

  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
      let dropCount = ((epoc - 1) * batchsize) `mod`  (length dataset)  -- 同じデータを使用しないようにするために前回の訓練で使用したところまでのデータを切り落とす。
          nowdataset = take batchsize (drop dropCount(cycle dataset))  -- 末尾に来たときに訓練データが減らないようにcycleにする。今回のデータセットをとってくる。
      let trainLossSum = sumTensors $ map (\(input,output) ->
                  let inputTensor = asTensor input
                      y = cbow model inputTensor
                      y' = unsqueeze (Dim 0) $ logSoftmax (Dim 0) y -- logSoftmaxで確率に落とし込む
                      outputTensor = asTensor [output]
                  in nllLoss' outputTensor y'
               ) nowdataset
          meanTrainLoss = trainLossSum / (asTensor [fromIntegral batchsize :: Float])
          lossTrainValue = (asValue meanTrainLoss)::Float 
      let validLossSum = sumTensors $ map (\(input,output) ->
                  let inputTensor = asTensor input
                      y = cbow model inputTensor
                      y' = unsqueeze (Dim 0) $ logSoftmax (Dim 0) y
                      outputTensor = asTensor [output]
                  in nllLoss' outputTensor y'
               ) validset
          meanValidLoss = validLossSum / validCount
          validLossValue = (asValue meanValidLoss)::Float
      showLoss 10 epoc validLossValue 
      u <- update model opt meanTrainLoss learningRate
      return (u, validLossValue)

  -- drawLearningCurve "Session6/result_graph/embedding4-2.png" "Learning Curve" [("Validation Loss",reverse losses)]


  -- Save params to use trained parameter in the next session
  -- trainedEmb :: Embedding
  let trainedEmb = embeddings trainedModel
  saveParams trainedEmb modelPath
  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- Load params
  initWordEmb <- makeIndependent $ zeros' [1]
  let initEmb = Embedding {wordEmbedding = initWordEmb}
  loadedEmb <- loadParams initEmb modelPath
  -- let testWord = B.pack $ encode "is"
  -- let testidx = wordToIndex testWord
  -- let weights = toDependent $ wordEmbedding loadedEmb
  -- let idxTensor = asTensor [testidx :: Int]
  -- let wordVec = embedding' weights idxTensor

  -- putStrLn $ "Word: " ++ show testWord ++ " -> index: " ++ show testidx ++ "\nEmbedding Vector: " ++ show wordVec

  let testWord1 = B.pack $ encode "it"
  let testWord2 = B.pack $ encode "this"
  let testidx1 = wordToIndex testWord1
  let testidx2 = wordToIndex testWord2
  let weights = toDependent $ wordEmbedding loadedEmb
  let idxTensor1 = asTensor [testidx1 :: Int]
  let idxTensor2 = asTensor [testidx2 :: Int]
  let wordVec1 = embedding' weights idxTensor1
  let wordVec2 = embedding' weights idxTensor2

  putStrLn $ "Word: " ++ show testWord1 ++ " -> index: " ++ show testidx1 ++ "\nEmbedding Vector: " ++ show wordVec1
  putStrLn $ "Word: " ++ show testWord2 ++ " -> index: " ++ show testidx2 ++ "\nEmbedding Vector: " ++ show wordVec2

  return ()