{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}

module IO
  ( readShot
  , datasetPathSource
  , loadVisor
  , trainC
  , datasetSampleSource
  ) where

import Types
import Visor
import Util
import Lib
import Static.Image
import Conduit
import System.FilePath
import System.Posix.Files
import System.Directory
import System.Random.Shuffle
import Data.Singletons.TypeLits
import Data.Proxy
import qualified Data.ByteString as BS
import Data.Serialize

readShot :: FilePath -> IO (Screenshot a)
readShot fp = do ebmp <- readRaw fp
                 case ebmp of
                   Left err -> error err
                   Right bmp -> return (Screenshot bmp)

datasetPathSource :: Dataset a -> RTSource FilePath
datasetPathSource set = sourceDirectoryDeep True (rootDir set) .| filterC ((== ".bmp") . takeExtension)

datasetSampleSource :: Dataset a -> RTSource (Screenshot a, LabelVec a)
datasetSampleSource set = datasetPathSource set .| shuffleC .| loadC
  where loadC = awaitForever$ \path -> do shot <- liftIO$ readShot path
                                          yield (shot, parseFilename set $ takeFileName path)

-- | Drain a source of its elements, and yield them in a random order
shuffleC :: RTConduit a a
shuffleC = do ls <- sinkList
              ls' <- liftIO$ shuffleM ls
              yieldMany ls'

loadVisor :: forall a.
  ( Creatable (Visor a)
  , GameState a
  ) => IO (Visor a)
loadVisor = do createDirectoryIfMissing True "data"
               exists <- fileExist path
               visor <- if exists then readVisor else newVisor
               return visor
  where
    name = symbolVal (Proxy :: Proxy (Title a))
    dir = "data"
    path = dir </> name

    readVisor :: IO (Visor a)
    readVisor = do bs <- BS.readFile path
                   case decode bs of
                     Left err -> (putStrLn$ "Error decoding visor: " ++ err) >> newVisor
                     Right v  -> return v

    newVisor :: IO (Visor a)
    newVisor = do putStrLn$ "Initializing new visor at " ++ path
                  return$ seeded 9

trainC :: ( GameState a
          , WVector (Widgets a)
          ) => Visor a -> RTConduit (Screenshot a, LabelVec a) (Visor a)
trainC visor =
  do ms <- await
     case ms of
       Nothing     -> return ()
       Just (x, y) -> do (v', ((p,c),l)) <- trainImage visor x y
                         liftIO.putStrLn$ "Correct: " ++ show p ++ "/" ++ show c ++ "\tLoss: " ++ show l
                         trainC v'
