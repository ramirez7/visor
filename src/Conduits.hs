{-# LANGUAGE TupleSections #-}

module Conduits ( module Conduit
                , module Conduits
                ) where

import Game
import Util
import Batch
import Visor
import System.Directory
import System.FilePath
import Data.Serialize
import qualified Data.ByteString as BS
import Data.Conduit.Zlib
import Conduit

-- | A source conduit that yields successive images from a dataset.
datasetSource :: Dataset -> IOSrc LabeledImage
datasetSource (Dataset root lFn rect wig dis) = paths .| pair
  where
    paths       = sourceDirectoryDeep True root .| filterC ((==".png") . takeExtension)
    imgSource p = do liftIO . putStrLn $ "Loading " ++ p
                     bs <- sourceFileBS p .| foldC
                     yield bs
    pair        = awaitForever $ \p -> imgSource p .| toRGB .| mapC (,lFn p)
    toRGB       = mapMC $ \bs -> loadImage bs rect wig dis

-- | Collect n LabeledImages and consolidate them into a single VBatch.
parseLabeledImage :: Visor -> Int -> IOConduit LabeledImage VBatch
parseLabeledImage visor n = interpret .| gatherC
  where
   interpret = mapC (toVBatch . features . game $ visor)
   gatherC = awaitForever $ \x -> do stacked <- takeC (n-1) .| foldlC (zipWith stack) x
                                     yield stacked

-- | Write VBatches to the directory specified
batchSink :: String -> IOSink VBatch
batchSink dirname = do liftIO $ createDirectoryIfMissing True dir
                       mapC encode .| iterWrite 0
  where
    dir = "data" </> "batch" </> dirname
    iterWrite :: Int -> IOSink BS.ByteString
    iterWrite i = do liftIO . putStrLn $ "Loading batch " ++ show i
                     mbs <- await
                     case mbs of
                       Just bs -> do yield bs .| compress 9 defaultWindowBits .| sinkFileBS (dir</>dirname ++ show i)
                                     liftIO . putStrLn $ "Wrote batch " ++ show i
                                     iterWrite (i+1)
                       Nothing -> do return ()
                                     liftIO . putStrLn $ "Pipeline exhausted, exiting"

-- | Convert a dataset into VBatches and write them to disk
genBatch :: Int -> Dataset -> Visor -> IO ()
genBatch n set visor = runConduitRes $ datasetSource set
                                    .| parseLabeledImage visor n
                                    .| batchSink (title . game $ visor)
