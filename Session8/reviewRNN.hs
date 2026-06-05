{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


module Main (main) where
-- module reviewRNN (main) where

import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import Data.Char (toLower)
import qualified Data.ByteString.Lazy as B
import qualified Data.Map.Strict as M 
import Data.List (elemIndex, nub)
import Data.Word (Word8)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Train (update, showLoss, sumTensors)
import Torch.Control (mapAccumM)
import Torch.Optim (GD(..))
import Torch.Tensor (Tensor, select, shape, asTensor, asValue)
import Torch.Functional (stack, Dim(..), mseLoss)
import Torch.Layer.RNN (RnnHypParams(..), RnnParams(..), rnnLayers)
import Torch.Layer.MLP (MLPHypParams(..), MLPParams, ActName(..), mlpLayer)
import Torch.Device (Device(..), DeviceType(..))
import ML.Exp.Chart (drawLearningCurve)
import Evaluation (accuracy, macroF1, weightedF1, confusionMatrix)

-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview

-- model
data ModelSpec = ModelSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

data Model = Model {
  emb :: Embedding,
  -- TODO: add RNN
  rnn :: RnnParams,  -- RNN
  mlp :: MLPParams   -- RNNの出力をスコア(1つの数字)に変換する層
} deriving (Generic, Parameterized)  -- Showを外した


instance
  Randomizable
    ModelSpec
    Model
  where
    sample ModelSpec {..} = do
        randEmb <- Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim])  -- ランダムな値のembeddingを作る
        -- TODO: add RNN initilization
        let rnnSpec = RnnHypParams {
            dev = Device CPU 0,
            bidirectional = False,  -- True if BiLSTM, False otherwise
            inputSize = wordDim,  -- The number of expected features in the input x
            hiddenSize = 256, -- The number of features in the hidden state h
            numLayers = 1,  -- Number of recurrent layers
            hasBias = True   -- If False, then the layer does not use bias weights b_ih and b_hh.
            }
        randRNN <- sample rnnSpec

        let mlpSpec = MLPHypParams (Device CPU 0) 256 [(1, Id)]
        randMLP <- sample mlpSpec

        return $ Model randEmb randRNN randMLP

-- randomize and initialize embedding with loaded params
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return randomizedModel {emb = loadedEmb {-, rnn = rnn randomizedModel -}}

-- ランダムな値のはいった初期値を作る
initializeRandom :: ModelSpec -> IO Model
initializeRandom modelSpec = sample modelSpec

-- 記号を判定する
isUnncessaryChar :: 
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", ",", "!", "?", "<br", "<", "/", ">", "(", ")", "*", "@", "-", "&", "#", ";", ":", "\\", "[", "]", "_"]
--リストに含まれた文字について、encode([String]を[Word8型]へ)したhead(前から順に。すなわち[Word8]->Word8)を持ってきて、strとelem(要素であるかどうか判断し)、Bool型を出力。

-- テキストを記号を排除した単語ごとのリスト[[B.ByteString]](行数*単語数)にする
preprocess ::
  String -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
-- 単語ごとに区切る　 ["I am a dog.", "I don't like dogs"]->[["I", "am", "a", "dog"], ["I", "don't", "like", "dogs"]]
  where
    lowerTexts = map toLower texts
    bstexts = B.pack (encode lowerTexts) -- String を ByteString に変換
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack bstexts)
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


-- 単語ベクトルのリストからスコアを推測する
-- 引数: -- Model, [Int] (レビューの単語IDのリスト)
-- 戻り値: Tensor (予測されたスコア)
forward :: Model -> [Int] -> Tensor
forward Model{..} wordIds =
  if null wordIds then zeros' [1, 1]
  else
    -- 文章を一単語ずつに分けてベクトル化する (Embedding)
    let
      weight = toDependent (wordEmbedding emb)                -- 重み行列から値を取り出せる状態にする
      wordVecs = map (\idx -> select 0 idx weight) wordIds    -- wordIds(例: [5, 12, 3]) の数字を使って、該当するベクトルを1つずつ抜き出す
      inputs = stack (Dim 0) wordVecs     -- バラバラのベクトルを縦に重ねて、1つのまとまったテンソル(行列)にする [文章の単語数, ベクトルの次元数(wordDim)] 

      -- 前から順にRNNに渡す

      h0 = zeros' [1, 256]  -- 最初の単語を読む前の「初期の記憶(h0)」として、ゼロで埋まったベクトルを用意する　[記憶レイヤー数, 単語ベクトル次元数]
      
      -- RNNに処理させる
      (allOutputs, finalOutput) = rnnLayers rnn Tanh Nothing h0 inputs  -- rnnLayers 引数: (RNNの重み) (活性化関数) (ドロップアウト) (初期値) (入力データ) 戻り値: (すべてのステップの出力, 最後の記憶)

      -- 最終的な値をMLPに渡してスコアを出す
      score = mlpLayer mlp (select 0 0 finalOutput)   -- RNNが最後まで読んで出した「最後の出力(finalOutput)」をMLPに通す
    in   score

-- your amazon review json
amazonReviewPathTrain :: FilePath
amazonReviewPathTrain = "Session8/data/train_light.jsonl"

amazonReviewPathValid :: FilePath
amazonReviewPathValid = "Session8/data/valid_light.jsonl"

amazonReviewPathTest :: FilePath
amazonReviewPathTest = "Session8/data/test.jsonl"

outputPath :: FilePath
outputPath = "Session8/data/review-texts-emb-2.txt"

embeddingPath =  "Session8/data/glove_emb.params"

wordLstPath = "Session8/data/glove_wordlst.txt"

grahpPath = "Session8/result_graph/reviewRNN-emb-2.png"

-- jsonをHaskellで使えるように
decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (head $ encode "\n") jsonl
  in sequenceA $ map eitherDecode jsonList  
  -- sequenceA:Evaluate each action in the structure from left to right, and collect the results.
  -- examples) sequenceA [Right 1, Right 2, Right 3] -> Right [1,2,3], sequenceA [Right 1, Right 2, Right 3, Left 4] -> Left 4

iter :: Int
iter = 100

learningRate :: Tensor
learningRate = asTensor (0.0001 :: Float) 

batchsize :: Int
batchsize = 16


main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPathTrain
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews
  jsonlValid <- B.readFile amazonReviewPathValid
  let amazonReviewsValid = decodeToAmazonReview jsonlValid
  let reviewsValid = case amazonReviewsValid of
                  Left err -> []
                  Right reviews -> reviews
  jsonlTest <- B.readFile amazonReviewPathTest
  let amazonReviewsTest = decodeToAmazonReview jsonlTest
  let reviewsTest = case amazonReviewsTest of
                  Left err -> []
                  Right reviews -> reviews

  -- load word list (It's important to use the same list as whan creating embeddings)
  rawWordlst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  let wordlst = filter (not . B.null) rawWordlst

  -- load params (set　wordDim　and wordNum same as session5)
  let wordLength = length wordlst
  let modelSpec = ModelSpec {
    wordDim = 300, 
    wordNum = wordLength
  }
  -- initModel <- initializeRandom modelSpec
  initModel <- initialize modelSpec embeddingPath

  putStrLn "モデル初期化OK"

  let wordToIndex = wordToIndexFactory wordlst
  let dataset = map (\rev -> 
          let processed = concat $ preprocess (text rev)
              wIds = map wordToIndex processed
              target = rating rev
          in (wIds, target)
        ) reviews
  let validset = map (\rev -> 
          let processed = concat $ preprocess (text rev)
              wIds = map wordToIndex processed
              target = rating rev
          in (wIds, target)
        ) reviewsValid
  let validCount = asTensor (fromIntegral (length validset) :: Float)

  ((trainedModel, _), losses) <- mapAccumM [1..iter] (initModel, GD) $ \epoc (model, opt) -> do
      let dropCount = ((epoc - 1) * batchsize) `mod` (length dataset)
          nowdataset = take batchsize (drop dropCount (cycle dataset))
      let trainLossSum = sumTensors $ map (\(wIds, targetRating) ->
              let predictedScore = forward model wIds
                  predictedScore2x = predictedScore * asTensor (2.0 :: Float)
                  targetTensor = asTensor [targetRating * 2.0 :: Float]
              in mseLoss targetTensor predictedScore2x 
           ) nowdataset
      let validLossSum = sumTensors $ map (\(wIds, targetRating) ->
              let predictedScore = forward model wIds
                  predictedScore2x = predictedScore * asTensor (2.0 :: Float)
                  targetTensor = asTensor [targetRating * 2.0 :: Float]
              in mseLoss targetTensor predictedScore2x
           ) validset
      let meanTrainLoss = trainLossSum / (asTensor (fromIntegral batchsize :: Float))
      let lossTrainValue = (asValue meanTrainLoss) :: Float 
      let meanValidLoss = validLossSum / validCount
      let lossValidValue = (asValue meanValidLoss) :: Float 
      showLoss 10 epoc lossValidValue 
      u <- update model opt meanTrainLoss learningRate
      return (u, lossValidValue)

  drawLearningCurve grahpPath "Learning Curve" [("Validation Loss",reverse losses)]
  
  let allResults = map (\rev -> 
          let processed = concat $ preprocess (text rev)
              wIds = map wordToIndex processed
              
              -- 未知語の数をカウント: wordToIndexが (length wordlst) を返したものが未知語
              oovCount = length $ filter (== length wordlst) wIds
              oovCountUni = length $ filter (== length wordlst) (nub wIds)
              totalWords = length wIds

              predicted = asValue (forward trainedModel wIds) :: Float
              actual = rating rev
              predictedInt = min 5 (max 1 (round predicted :: Int))
              actualInt = round actual :: Int
              
              isCorrect = predictedInt == actualInt
              resultText ="【正解】: " ++ show actualInt ++ " | 【予測】: " ++ show predictedInt ++ " (" ++ show predicted ++ ") | OOV: " ++ show oovCount ++ "/" ++ show totalWords ++ " | 【本文】: " ++ text rev
          in (actualInt, predictedInt, oovCount, oovCountUni, totalWords, resultText)
        ) reviewsTest

  -- リストを分解して評価用のデータを作る
  let actualInts = map (\(a,_,_,_,_,_) -> a) allResults
  let predictedInts = map (\(_,p,_,_,_,_) -> p) allResults
  let outputText = unlines (map (\(_,_,_,_,_,txt) -> txt) allResults)
  
  -- OOVの統計
  let totalOOV = sum $ map (\(_,_,o,_,_,_) -> o) allResults
  let totalOOVwords = sum $ map (\(_,_,_,w,_,_) -> w) allResults
  let totalTestWords = sum $ map (\(_,_,_,_,t,_) -> t) allResults
  let oovRate = (fromIntegral totalOOV / fromIntegral totalTestWords) * 100 :: Float
  let oovRatewords = (fromIntegral totalOOVwords / fromIntegral wordLength) * 100 :: Float

  -- Evaluationモジュールを使った計算
  let classes = [1, 2, 3, 4, 5]
  let accRate = Evaluation.accuracy actualInts predictedInts * 100
  let macF1 = Evaluation.macroF1 classes actualInts predictedInts
  let weiF1 = Evaluation.weightedF1 classes actualInts predictedInts
  let cMat = Evaluation.confusionMatrix classes actualInts predictedInts

  -- 結果の出力
  putStrLn $ "\n===== 評価結果 ====="
  putStrLn $ "【正解率 (Accuracy)】: " ++ show accRate ++ " %"
  putStrLn $ "【Macro F1スコア】   : " ++ show macF1
  putStrLn $ "【Weighted F1スコア】: " ++ show weiF1
  putStrLn $ "【未知語率】  : " ++ show totalOOV ++ " / " ++ show totalTestWords ++ " (" ++ show oovRate ++ " %)"
  putStrLn $ "【未知語彙率】  : " ++ show totalOOVwords ++ " / " ++ show wordLength ++ " (" ++ show oovRatewords ++ " %)"
  
  putStrLn "\n【Confusion Matrix】"
  putStrLn "      予測1 予測2 予測3 予測4 予測5"
  mapM_ (\(r, row) -> putStrLn $ "正解" ++ show r ++ ": " ++ show row) (zip classes cMat)


  -- let outputText = unlines (map snd allResults)
  writeFile outputPath outputText

  return ()