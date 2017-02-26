{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}

module Main where

import Types
import Visor
import IO
import Lib
import Games.Melee
import System.Environment
import Conduit

type Game = Melee
type BatchSize = 10

set :: Dataset Melee
set = dataset

main :: IO ()
main = getArgs >>= main'

main' :: [String] -> IO ()
main' ["parse", path] = print l
  where l = parseFilename set path

main' ["ls"] =
  do paths <- runConduitRes$ datasetPathSource set .| sinkList
     putStrLn$ unlines paths
     putStrLn$ show (length paths) ++ " images in dataset"

main' ["label", path] =
  do v :: Visor Game <- loadVisor
     shot <- readShot path
     y <- feedImage shot v
     putStrLn$ "Label: " ++ show y
     putStrLn$ "Interpretation: " ++ show (delabel y :: Game)

main' ["train"] =
  do v :: Visor Game <- loadVisor
     Just v' <- runConduitRes$ datasetSampleSource set False .| trainC v .| takeC 10 .| lastC
     saveVisor v'

main' ["makebatch"] =
  do clear "batch"
     runConduitRes$ datasetSampleSource set True
                 .| batchify
                 .| (saveMany "batch" :: RTSink (BatchVec BatchSize Melee))

main' ["trainbatch"] =
  do v :: Visor Game <- loadVisor
     Just v' <- runConduitRes$ (loadMany "batch" :: RTSource (BatchVec BatchSize Melee))
                            .| trainBatchC v
                            .| takeC 10
                            .| lastC
     saveVisor v'

main' l = putStrLn$ "Unrecognized argument list: " ++ unwords l
