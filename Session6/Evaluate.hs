{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where
-- module Evaluate (main) where

import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import Data.Maybe (mapMaybe)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BC
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import Data.Char (toLower)
import Text.Read (readMaybe)

import Torch.DType (DType(..))
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', logSoftmax, nllLoss', sumDim,sumAll, Dim(..), KeepDim(..),unsqueeze)
import Torch.NN (Parameterized(..), Parameter)
import Torch.Serialize (loadParams)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Layer.MLP    (MLPParams, MLPHypParams(..),ActName(..),mlpLayer)
import Torch.TensorFactories (eye')

textFilePath = "Session6/data/answer-answer.test.tsv"
modelPath =  "Session6/data/sample_embedding4-2.params"
wordLstPath = "Session6/data/sample_wordlst2.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int,
  wordDim :: Int
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

data Model = Model {
    mlp :: MLPParams,
    embeddings :: Embedding
  } deriving (Generic, Parameterized)

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", ",", "!", "?", "<br", "<", "/", ">", "(", ")", "*", "@", "-", "&", "#", ";", ":", "\\", "[", "]", "_"]

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

toyEmbedding :: EmbeddingSpec -> Tensor
toyEmbedding EmbeddingSpec{..} = eye' wordNum wordDim

parseData :: B.ByteString -> [(Float, B.ByteString, B.ByteString)]
parseData text =
    let textLines = BC.split '\n' text
        getPair line = case BC.split '\t' line of
            (scoreStr : s1 : s2 : _) ->
                case readMaybe (BC.unpack scoreStr) :: Maybe Float of
                    Just score -> Just (score, s1, s2)
                    Nothing -> Nothing
            _ -> Nothing
    in mapMaybe getPair textLines

cosSim :: Tensor -> Tensor -> Float
cosSim vec1 vec2 =
    let ab = asValue (sumAll (vec1 * vec2)) :: Float
        norma = sqrt (asValue (sumAll (vec1 * vec1)) :: Float)
        normb = sqrt (asValue (sumAll (vec2 * vec2)) :: Float)
    in ab / (norma * normb)

main :: IO ()
main = do
  
  wordData <- B.readFile wordLstPath
  let wordlst = BC.split '\n' wordData
      wordToIndex = wordToIndexFactory wordlst

  let mywordDim = 9
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst + 1, wordDim = mywordDim}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding { wordEmbedding = wordEmb }
  
  loadedEmb <- loadParams emb modelPath
  let weights = toDependent $ wordEmbedding loadedEmb

  texts <- B.readFile textFilePath
  let lowerTexts = BC.map toLower texts
  let parseText = parseData lowerTexts

  let process (score, s1, s2) = do
        let wordLines1 = head $ preprocess s1
            wordLines2 = head $ preprocess s2
            idxs1 = map wordToIndex wordLines1
            idxs2 = map wordToIndex wordLines2
            -- unkId = length wordlst
            -- idxs1' = filter (/= unkId) idxs1
            -- idxs2' = filter (/= unkId) idxs2
            indexlst1 = asTensor [idxs1 :: [Int]]
            indexlst2 = asTensor [idxs2 :: [Int]]
            sumvec1 = sumDim (Dim 1) RemoveDim Float (embedding' weights indexlst1)
            sumvec2 = sumDim (Dim 1) RemoveDim Float (embedding' weights indexlst2)
            
            sim = cosSim sumvec1 sumvec2
            
            score5 = if sim >= 0.98 then 5.0
                     else if sim >= 0.95 then 4.0
                     else if sim >= 0.85 then 3.0
                     else if sim >= 0.70 then 2.0
                     else if sim >= 0.5 then 1.0
                     else 0.0
                     
            diff = abs (score5 - score)
                     
        putStrLn $ "Similarity: " ++ show score5 ++ " Score: " ++ show score ++ " Difference: " ++ show diff
        -- putStrLn $ show sumvec1
        -- putStrLn $ show sumvec2
        putStrLn $ show sim
        return diff

  differences <- mapM process parseText

  let meanDiff = (sum differences) / (fromIntegral (length parseText))
  putStrLn $ " Diff_Mean: " ++ show meanDiff

  let wordA = B.pack $ encode "game"
      wordB = B.pack $ encode "games"
  
  let idxA = wordToIndex wordA
      idxB = wordToIndex wordB
  
  let vecA = embedding' weights (asTensor [idxA :: Int])
      vecB = embedding' weights (asTensor [idxB :: Int])
  
  let simScore = cosSim vecA vecB
  
  putStrLn "\n--- Word Embedding Similarity Test ---"
  putStrLn $ "Word A: " ++ show wordA ++ " (index: " ++ show idxA ++ ")" ++ " (vec: " ++ show vecA ++ ")"
  putStrLn $ "Word B: " ++ show wordB ++ " (index: " ++ show idxB ++ ")" ++ " (vec: " ++ show vecA ++ ")"
  putStrLn $ "Cosine Similarity: " ++ show simScore

  return ()