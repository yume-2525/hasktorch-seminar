{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

module Main (main) where

import Codec.Binary.UTF8.String (encode)
import qualified Data.ByteString.Lazy as B
import qualified Data.Map.Strict as M
import Data.List (sortBy)
import Data.Ord (comparing)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Tensor (Tensor, select, asValue)
import Torch.Functional (mul, sumAll)
import Torch.Device (Device(..), DeviceType(..))

-- モデル構造
data ModelSpec = ModelSpec {
  wordNum :: Int,
  wordDim :: Int
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

instance Randomizable ModelSpec Embedding where
    sample ModelSpec {..} = Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim])

-- パラメータの読み込み関数
initializeEmb :: ModelSpec -> FilePath -> IO Embedding
initializeEmb spec path = do
  randEmb <- sample spec
  loadParams randEmb path

-- コサイン類似度（2つのベクトルの向きの近さを -1.0 〜 1.0 で計算）
cosSim :: Tensor -> Tensor -> Float
cosSim vec1 vec2 =
    let ab = asValue (sumAll (vec1 * vec2)) :: Float
        norma = sqrt (asValue (sumAll (vec1 * vec1)) :: Float)
        normb = sqrt (asValue (sumAll (vec2 * vec2)) :: Float)
    in ab / (norma * normb)

-- ファイルパス設定（ご自身の環境に合わせてください）
embeddingPath = "Session8/data/glove_emb.params"
wordLstPath = "Session8/data/glove_wordlst.txt"

main :: IO ()
main = do
  -- 1. 辞書の読み込み
  rawWordlst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  let wordlst = filter (not . B.null) rawWordlst
  let wordLength = length wordlst

  -- 2. GloVeパラメータの読み込み
  let modelSpec = ModelSpec { wordDim = 300, wordNum = wordLength }
  emb <- initializeEmb modelSpec embeddingPath
  let weight = toDependent (wordEmbedding emb)

  putStrLn "GloVeの読み込み完了"

  -- 単語からIDを取得する関数
  let wordToIndex = M.fromList (zip wordlst [0..])
  let getIdx w = case M.lookup w wordToIndex of
                   Just idx -> idx
                   Nothing  -> error $ "単語が見つかりません: " ++ show w
  
  -- 3. 対象となる3つの単語のベクトルを取得
  let vecbreakfast  = select 0 (getIdx "breakfast") weight
  let vecmorning   = select 0 (getIdx "morning") weight
  let vecevening = select 0 (getIdx "evening") weight

  -- 4. ベクトルの演算: breakfast - morning + evening
  let targetVec = vecbreakfast - vecmorning + vecevening

  -- 5. すべての単語ベクトルとの類似度を計算
  let similarities = map (\(idx, word) -> 
          let v = select 0 idx weight
              sim = cosSim targetVec v
          in (B.unpack word, sim) -- 表示用に文字列化("queen", 0.88415766)
        ) (zip [0..] wordlst) -- [(0, "the"), (1, "of"), (2, "and"), ...]を作る

  -- 6. 類似度が高い順に並び替える
  let sorted = reverse $ sortBy (comparing snd) similarities

  -- 7. 上位10件を表示
  putStrLn "\n===== 結果トップ10 ====="
  mapM_ (\(w, sim) -> putStrLn $ "単語: " ++ show (map (toEnum . fromEnum) w :: String) ++ "\t\t| 類似度: " ++ show sim) (take 10 sorted)