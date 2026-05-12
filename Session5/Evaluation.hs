module Evaluation (
    accuracy,
    precision,
    recall,
    confusionMatrix,
    macroF1,
    microF1,
    weightedF1,
) where


tp :: [Int] -> [Int] -> Int
tp actual predicted = length $ filter (\ x -> x == (1,1)) $ zip actual predicted

fp :: [Int] -> [Int] -> Int
fp actual predicted = length $ filter (\ x -> x == (0,1)) $ zip actual predicted

tn :: [Int] -> [Int] -> Int
tn actual predicted = length $ filter (\ x -> x == (0,0)) $ zip actual predicted

fn :: [Int] -> [Int] -> Int
fn actual predicted = length $ filter (\ x -> x == (1,0)) $ zip actual predicted


accuracy :: [Int] -> [Int] -> Float
accuracy actual predicted = (fromIntegral $ (tp actual predicted) + (tn actual predicted)) /(fromIntegral $ length actual)

precision :: [Int] -> [Int] -> Float
precision actual predicted = (fromIntegral $ (tp actual predicted)) / (fromIntegral $ (tp actual predicted) + (fp actual predicted))

recall :: [Int] -> [Int] -> Float
recall actual predicted = (fromIntegral $ (tp actual predicted)) / (fromIntegral $ (tp actual predicted) + (fn actual predicted))


confusionMatrix :: [Int] -> [Int] -> [[Int]]
confusionMatrix actual predicted = [[(tp actual predicted), (tn actual predicted)],[(fp actual predicted), (fn actual predicted)]]


f1_class1 :: [Int] -> [Int] -> Float
f1_class1 actual predicted = (2.0 * prec1 * reca1) / (prec1 + reca1)
    where 
        prec1 = precision actual predicted
        reca1 = recall actual predicted

f1_class0 :: [Int] -> [Int] -> Float
f1_class0 actual predicted = (2.0 * prec0 * reca0) / (prec0 + reca0)
    where
        prec0 = (fromIntegral $ (tn actual predicted)) / (fromIntegral $ (tn actual predicted) + (fn actual predicted))
        reca0 = (fromIntegral $ (tn actual predicted)) / (fromIntegral $ (tn actual predicted) + (fp actual predicted))


macroF1 :: [Int] -> [Int] -> Float
macroF1 actual predicted = ((f1_class1 actual predicted) +(f1_class0 actual predicted)) / 2.0

weightedF1 :: [Int] -> [Int] -> Float
weightedF1 actual predicted = f1_class1' * rate_class1 + f1_class0' * rate_class0
    where
        f1_class1' = f1_class1 actual predicted
        f1_class0' = f1_class0 actual predicted
        len_class1 = fromIntegral $ length $ filter (\x -> x == 1) actual
        len_class0 = fromIntegral $ length $ filter (\x -> x == 0) actual
        len = fromIntegral $ length actual
        rate_class1 = len_class1 / len
        rate_class0 = len_class0 / len

microF1 :: [Int] -> [Int] -> Float
microF1 actual predicted = accuracy actual predicted