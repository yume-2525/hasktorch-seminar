{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

-- docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/hasktorch-nlp-introduction && stack run day6-parse"

-- module Main (main) where
module Parser (main) where
-- json
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics

data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

-- jsonとの変換を可能にする
instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,           -- スコア
  title :: String,           -- レビューのタイトル
  text :: String,            -- レビュー本文
  images :: [Image],         -- 画像のリスト
  asin :: String,            -- 商品ID
  parent_asin :: String,     -- 親商品のID
  user_id :: String,         -- ユーザーID
  timestamp :: Int,          -- 投稿日時
  verified_purchase :: Bool, -- 購入確認済みか (True/False)
  helpful_vote :: Int        -- 「役に立った」の数
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "Session7/data/train.jsonl"

-- ファイルからリストに変換する
decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl     -- 改行で区切ってリストにする
  in sequenceA $ map eitherDecode jsonList      -- sequenceA：全部成功(Right)したらリストにする　　　eitherDecode:jsonからHaskellの型にする。

main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  case amazonReviews of
    Left err -> print err
    Right reviews -> print reviews