module Evaluation (
    accuracy,
    precision,
    recall,
    f1,
    confusionMatrix,
    macroF1,
    microF1,
    weightedF1
) where

-- 特定のクラス(c)に対する TP, FP, FN を計算する
tp :: Int -> [Int] -> [Int] -> Int
tp c actual predicted = length $ filter (\(a, p) -> a == c && p == c) $ zip actual predicted

fp :: Int -> [Int] -> [Int] -> Int
fp c actual predicted = length $ filter (\(a, p) -> a /= c && p == c) $ zip actual predicted

fn :: Int -> [Int] -> [Int] -> Int
fn c actual predicted = length $ filter (\(a, p) -> a == c && p /= c) $ zip actual predicted

-- 全体のAccuracy（単純な正解率）
accuracy :: [Int] -> [Int] -> Float
accuracy actual predicted = 
    let correct = length $ filter (\(a, p) -> a == p) $ zip actual predicted
    in fromIntegral correct / fromIntegral (length actual)

-- 特定のクラス(c)に対する Precision, Recall, F1
precision :: Int -> [Int] -> [Int] -> Float
precision c actual predicted = 
    let tp_val = fromIntegral (tp c actual predicted)
        fp_val = fromIntegral (fp c actual predicted)
    in if tp_val + fp_val == 0 then 0.0 else tp_val / (tp_val + fp_val)

recall :: Int -> [Int] -> [Int] -> Float
recall c actual predicted = 
    let tp_val = fromIntegral (tp c actual predicted)
        fn_val = fromIntegral (fn c actual predicted)
    in if tp_val + fn_val == 0 then 0.0 else tp_val / (tp_val + fn_val)

f1 :: Int -> [Int] -> [Int] -> Float
f1 c actual predicted = 
    let p = precision c actual predicted
        r = recall c actual predicted
    in if p + r == 0 then 0.0 else (2.0 * p * r) / (p + r)

-- 任意のサイズ（5x5など）の混同行列を作成する
confusionMatrix :: [Int] -> [Int] -> [Int] -> [[Int]]
confusionMatrix classes actual predicted =
    [ [ cell r c | c <- classes ] | r <- classes ]
  where 
    -- 行r(正解), 列c(予測) の数をカウント
    cell r c = length $ filter (\(a, p) -> a == r && p == c) (zip actual predicted)

-- Macro F1 (全クラスのF1の単純平均)
macroF1 :: [Int] -> [Int] -> [Int] -> Float
macroF1 classes actual predicted = 
    let f1s = map (\c -> f1 c actual predicted) classes
    in sum f1s / fromIntegral (length classes)

-- Weighted F1 (データ数で重み付けしたF1の平均)
weightedF1 :: [Int] -> [Int] -> [Int] -> Float
weightedF1 classes actual predicted = 
    let total = fromIntegral (length actual)
        weight c = fromIntegral (length $ filter (== c) actual) / total
        f1s = map (\c -> f1 c actual predicted * weight c) classes
    in sum f1s

-- Micro F1 (多クラス分類では全体Accuracyと一致する)
microF1 :: [Int] -> [Int] -> Float
microF1 actual predicted = accuracy actual predicted