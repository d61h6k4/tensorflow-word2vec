{-# LANGUAGE OverloadedLists #-}
module Main where

import Control.Applicative (ZipList(..))
import Data.Traversable (sequenceA)
import Data.Int (Int32, Int64)
import Data.List (genericLength, tails)
import Data.Text (Text)
import Data.Vector (Vector)

import qualified NLP.Corpora.Dictionary as Dictionary
import qualified Data.Text.IO as Text
import qualified Data.Map.Strict as Map
import qualified Data.Vector as Vector

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Types as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF (svd, slice)

import TensorFlow.Examples.Text8.InputData
import TensorFlow.Examples.Text8.Parse

data Model = Model
  { evaluate :: TF.TensorData Int32 -> TF.TensorData Float -> TF.Session (Vector Float)
  }

numberOfTokens :: Int32
numberOfTokens = 10000

esvd :: TF.Build Model
esvd = do
  indices <- TF.placeholder [-1, 2]
  values <- TF.placeholder [-1]
  occurencesMatrix <-
    TF.render
      (TF.sparseToDense
         indices
         (TF.vector [numberOfTokens, numberOfTokens])
         values
         (TF.scalar 0))
  embeddings <-
    TF.render
      (let (s, u, v) = TF.svd occurencesMatrix
           uPart =
             TF.slice u (TF.vector [0, 0]) (TF.vector [numberOfTokens, 50])
           vPart =
             TF.slice v (TF.vector [0, 0]) (TF.vector [numberOfTokens, 50])
       in TF.add uPart vPart)
  return
    Model
    { evaluate =
        \indicesFeed valuesFeed ->
          TF.runWithFeeds
            [TF.feed indices indicesFeed, TF.feed values valuesFeed]
            embeddings
    }

getSparseOccurencesMatrix :: [[Int]] -> ([Vector Int], [Float])
getSparseOccurencesMatrix =
  unzip . Map.toList . Map.fromListWith (+) . map (flip (,) 1 . Vector.fromList)


-- | Sliding window function.
-- Implementation took from <SO http://stackoverflow.com/questions/27726739/implementing-an-efficient-sliding-window-algorithm-in-haskell>
window :: Int -> [a] -> [[a]]
window size = transpose' . take size . tails


transpose' :: [[a]] -> [[a]]
transpose' = getZipList . sequenceA . map ZipList

main :: IO ()
main = do
  text8tokens' <- readText8Tokens =<< text8Data
  let text8tokens = (take 20000 text8tokens')
  dictionary <-
    Dictionary.filterExtremes 0 (fromIntegral numberOfTokens - 2) =<<
    Dictionary.addDocument text8tokens =<< Dictionary.new
  (indices, values) <-
    fmap (getSparseOccurencesMatrix . window 2) (
    mapM
      (\token ->
         Dictionary.get token dictionary >>=
         return . maybe ((fromIntegral numberOfTokens) - 1) id)
      text8tokens)
  wordVectors <-
    TF.runSession
      (do let indicesTD =
                TF.encodeTensorData
                  [genericLength indices, 2]
                  (fromIntegral <$> mconcat indices)
              valuesTD =
                TF.encodeTensorData
                  [genericLength indices]
                  (Vector.fromList values)
          model <- TF.build esvd
          evaluate model indicesTD valuesTD)
  print (Vector.length wordVectors)
  mapM_ print (Vector.take 10 wordVectors)
