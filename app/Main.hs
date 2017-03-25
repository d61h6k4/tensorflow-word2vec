{-# LANGUAGE OverloadedLists #-}
module Main where

import Data.Int (Int32, Int64)
import Data.List (genericLength)
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
  uMatrix <-
    TF.render
      (let (s, u, v) = TF.svd occurencesMatrix
       in u)
  embeddings <-
    TF.render (TF.slice uMatrix (TF.vector [(0 :: Int32), 0]) (TF.vector [-1, 50]))
  return
    Model
    { evaluate =
        \indicesFeed valuesFeed ->
          TF.runWithFeeds
            [TF.feed indices indicesFeed, TF.feed values valuesFeed]
            embeddings
    }

getSparseOccurencesMatrix :: [Text] -> Dictionary.Dictionary -> IO ([Vector Int], [Float])
getSparseOccurencesMatrix text8tokens dictionary =
  return . unzip . Map.toList . Map.fromListWith (+) =<<
  mapM
    (\(token, token') -> do
       tid <- Dictionary.get token dictionary
       tid' <- Dictionary.get token' dictionary
       let token_x = maybe unkId id tid
           token_y = maybe unkId id tid'
       return (Vector.fromList [token_x, token_y], 1)
       return (Vector.fromList [token_y, token_x], 1))
    (zip text8tokens (tail text8tokens))
  where
    unkId = (fromIntegral numberOfTokens ) - 1


main :: IO ()
main = do
  text8tokens <- readText8Tokens =<< text8Data
  dictionary <-
    Dictionary.filterExtremes 0 ((fromIntegral numberOfTokens) - 2) =<<
    Dictionary.addDocument text8tokens =<< Dictionary.new
  (indices, values) <- getSparseOccurencesMatrix text8tokens dictionary
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
          embeddings <- evaluate model indicesTD valuesTD
          return embeddings)
  print (Vector.length wordVectors)
  mapM_ print (Vector.take 5 wordVectors)
