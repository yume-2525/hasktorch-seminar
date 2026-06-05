{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

-- module Main (main) where
module LoadGlove (main) where

import GHC.Generics
import qualified Data.ByteString.Lazy.Char8 as BC
import Text.Read (readMaybe)
import Torch.Autograd (makeIndependent, toDependent)
import Torch.NN (Parameterized(..), Parameter)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (zeros')
import Torch.Functional (stack, Dim(..), cat)

-- Session 6 と同じ Embedding の型定義
data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

-- ファイルパスの設定
gloveTxtPath = "Session8/data/glove_top10k.txt"      -- さっきターミナルで作ったファイル
outModelPath = "Session8/data/glove_emb.params"      -- 出力するHasktorch用重みファイル
outWordLstPath = "Session8/data/glove_wordlst.txt"   -- 出力する単語リスト

-- GloVeの1行(ByteString)から、(単語, ベクトル)を取り出す関数
parseGloveLine :: BC.ByteString -> (BC.ByteString, [Float])
parseGloveLine line = 
  let parts = BC.words line
      word = head parts
      -- 残りの文字列をFloatに変換
      vecStr = tail parts
      vecFloat = map (\s -> case readMaybe (BC.unpack s) :: Maybe Float of
                              Just val -> val
                              Nothing  -> 0.0) vecStr
  in (word, vecFloat)

main :: IO ()
main = do
  gloveData <- BC.readFile gloveTxtPath
  
  -- 1行ずつパースする (空行は取り除く)
  let linesData = filter (not . BC.null) (BC.lines gloveData)
      parsedData = map parseGloveLine linesData
      
      -- 単語のリストと、ベクトルのリストに分ける
      wordlst = map fst parsedData
      vecs = map snd parsedData
      
  putStrLn $ "Loaded " ++ show (length wordlst) ++ " words."

  -- リストのベクトルたちをTensorにして、Dim 0方向で積み上げる (stack)
  -- 10,000語の50次元なら、shapeは [10000, 50] になる
  let tensorVecs = map asTensor vecs
      gloveTensor = stack (Dim 0) tensorVecs

  let oovVector = zeros' [1, 300]
  let combinedTensor = cat (Dim 0) [gloveTensor, oovVector]

  -- Session 6 と同じように Parameter に変換して Embedding 型に入れる
  wordEmbParam <- makeIndependent combinedTensor
  let embToSave = Embedding { wordEmbedding = wordEmbParam }

  -- ファイルに保存
  saveParams embToSave outModelPath
  BC.writeFile outWordLstPath (BC.unlines wordlst)