{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}

module Main where

import Types
import Visor
import Lib
import IO
import Games.Melee
import Games.Melee.Conduits
import Games.Melee.Graphic
import System.Environment
import Control.Monad
import Conduit
import System.FilePath
import Buffer

type Game = Melee
type BatchSize = 16

main :: IO ()
main = getArgs >>= main'

main' :: [String] -> IO ()
main' ["parse", path] = print l
  where l = parse (Path path :: Path Game)

main' ["ls"] =
  do paths :: [Path Game] <- runConduitRes$ pathSource .| sinkList
     putStrLn.unlines$ show <$> paths
     putStrLn$ show (length paths) ++ " images in dataset"

main' ["lsparse"] =
  do paths :: [Path Game] <- runConduitRes$ pathSource .| sinkList
     putStrLn.unlines$ showAndLabel <$> paths
     putStrLn$ show (length paths) ++ " images in dataset"

main' ["label"] =
  do v <- loadVisor
     runConduitRes$ pathSource .| shuffleC .| mapMC (liftIO . f v) .| sinkNull
       where
         f :: Visor Game -> Path Game -> IO ()
         f v p = do fx <- readShot p >>= feedImage v
                    if fx == parse p
                       then putStrLn "-"
                       else do print p
                               putStrLn$ "Expected: " ++ show (parse p)
                               putStrLn$ "Actual:   " ++ show fx

main' ["label", path] =
  do v :: Visor Game <- loadVisor
     y <- readShot (Path path) >>= feedImage v
     putStrLn$ "Label: " ++ show y
     putStrLn$ "Interpretation: " ++ show (delabel y :: Game)

main' ["crops", path] =
  do crops :: InputVec Game <- readShot (Path path :: Path Game) >>= extract
     clear "crops"
     dumpCrops 0 (dir</>"crops") crops

main' ["train"] =
  do v :: Visor Game <- loadVisor
     Just v' <- runConduitRes$ forever (datasetSampleSource True) .| trainC v .| takeC 100 .| lastC
     saveVisor v'

main' ["makebatch"] =
  do clear "batch"
     runConduitRes$ datasetSampleSource True
                 .| (batchify :: BatchC BatchSize Game)
                 .| saveMany "batch"

main' ["trainbatch"] =
  do v :: Visor Game <- loadVisor
     runConduitRes$ forever (loadMany "batch" :: RTSource (BatchVec BatchSize Game))
                 .| trainBatchC v
                 .| saveVisorContinuous


main' ["trainbatch",n] =
  do v :: Visor Game <- loadVisor
     Just v' <- runConduitRes$ forever (loadMany "batch" :: RTSource (BatchVec BatchSize Game))
                            .| trainBatchC v
                            .| takeC (read n)
                            .| lastC
     saveVisor v'

main' ["watch", read' -> x, read' -> y, read' -> w, read' -> h] =
  do v :: Visor Game <- loadVisor
     runConduitRes$ screenShotSource x y w h
                 .| mapMC (feedImage v)
                 .| mapC delabel
                 .| bufferedFilter 5
                 .| ioC (putStrLn . pretty)
                 .| collectGame
                 .| mapMC (liftIO . gameGraph)
                 .| sinkNull

main' ["kernels"] =
  do v :: Visor Game <- loadVisor
     saveKernels v

main' l = putStrLn$ "Unrecognized argument list: " ++ unwords l
