
module TensorFlow.Examples.Text8.Parse where

import Data.Text (Text)
import Path (parseRelFile)
import Path.IO (resolveFile')
import Codec.Archive.Zip (mkEntrySelector, withArchive, getEntry)
import Data.Vector (Vector)

import qualified Data.Text.Encoding as Text
import qualified Data.ByteString.Char8 as BSC8


readText8Tokens :: FilePath -> IO [Text]
readText8Tokens path = do
  resolvedPath <- resolveFile' path
  text8EntrySelector <- parseRelFile "text8" >>= mkEntrySelector
  tokens <- withArchive resolvedPath (getEntry text8EntrySelector)
  return (map Text.decodeUtf8 (BSC8.words tokens))
