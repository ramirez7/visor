{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}

module IO
  ( readShot
  , loadVisor
  , screenShotSource
  , dir
  , saveVisor
  , saveVisorContinuous
  , saveKernels
  , loadMany
  , ioC
  , pathSource
  , pmap
  , saveMany
  , deleteVisor
  , batchify
  , trainC
  , trainBatchC
  , datasetSampleSource
  , clear
  , shuffleC
  ) where

import Types
import Lib
import Visor
import Vector
import Util
import Network
import Layers.Convolution
import qualified Static.Image as I
import Conduit
import Control.Monad
import System.FilePath
import System.Posix.Files
import System.Directory
import System.Random.Shuffle
import System.Process
import Data.Array.Repa.IO.BMP
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Proxy
import qualified Data.ByteString as BS
import Data.Serialize
import Numeric

readShot :: Path a -> IO (Screenshot a)
readShot (Path fp) = do ebmp <- I.readRaw fp
                        case ebmp of
                          Left err -> error$ err ++ " " ++ fp
                          Right bmp -> return (Screenshot bmp)

screenShotSource :: Int -> Int -> Int -> Int -> RTSource (Screenshot a)
screenShotSource x y w h = forever$ liftIO takeshot >>= yield
 where
   cmd = "screencapture -xm -R" ++ show x ++ ',':show y ++ ',':show w ++ ',':show h ++ " -t bmp out.bmp"
   takeshot = do _ <- system cmd
                 Right img <- readImageFromBMP "out.bmp"
                 removeFile "out.bmp"
                 return (Screenshot img)

pathSource :: forall a. GameState a => RTSource (Path a)
pathSource = sourceDirectoryDeep True (unpath (rootDir :: Path a))
          .| filterC ((== ".bmp") . takeExtension)
          .| mapC Path

ioC :: (a -> IO ()) -> RTConduit a a
ioC f = awaitForever$ \x -> do liftIO$ f x
                               yield x

datasetSampleSource :: GameState a => Bool -> RTSource (Screenshot a, LabelVec a)
datasetSampleSource shuf = pathSource
                        .| (if shuf then shuffleC else awaitForever yield)
                        .| loadC

loadC :: GameState a => RTConduit (Path a) (Screenshot a, LabelVec a)
loadC = awaitForever$ \path -> do shot <- liftIO$ readShot path
                                  yield (shot, parse $ pmap takeFileName path)



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
    path = dir </> name

    readVisor :: IO (Visor a)
    readVisor = do bs <- BS.readFile path
                   case decode bs of
                     Left err -> error err
                     Right v  -> return v

    newVisor :: IO (Visor a)
    newVisor = do putStrLn$ "Initializing new visor at " ++ path
                  return$ seeded 9

saveVisor :: forall a.
  ( Serialize (Visor a)
  , GameState a
  ) => Visor a -> IO ()
saveVisor v = do exists <- fileExist path
                 flag <- if exists then do putStrLn$ "Visor found at " ++ path ++ ", delete?[Yn] "
                                           a <- getLine
                                           return$ a `notElem` ["n", "N"]
                                   else return True
                 when flag $ BS.writeFile path (encode v)
  where
    name = symbolVal (Proxy :: Proxy (Title a))
    path = dir </> name

saveVisorContinuous :: forall a.
  ( Serialize (Visor a)
  , GameState a
  ) => RTSink (Visor a)

saveVisorContinuous = awaitForever $ liftIO . BS.writeFile path . encode
  where
    name = symbolVal (Proxy :: Proxy (Title a))
    path = dir </> name

deleteVisor :: forall a p.
  ( GameState a
  ) => p a -> IO ()
deleteVisor _ = removeFile path
  where
    name = symbolVal (Proxy :: Proxy (Title a))
    path = dir </> name

trainC :: ( GameState a
          , WVector (Widgets a)
          ) => Visor a -> RTConduit (Screenshot a, LabelVec a) (Visor a)
trainC visor =
  do ms <- await
     case ms of
       Nothing     -> return ()
       Just (x, y) ->
         do (v', ((p,c),l)) <- trainImage visor x y
            liftIO.putStrLn$ showString "Correct: "
                           . shows p
                           . showString "/"
                           . shows c
                           . showString "\tLoss: "
                           . showEFloat (Just 5) l $""
            yield v'
            trainC v'

trainBatchC :: ( Stack n (Widgets a)
               ) => Visor a -> RTConduit (BatchVec n a) (Visor a)

trainBatchC (Visor visor) = go visor [] []
  where
    go v ls ps =
      do mb <- await
         case mb of
           Nothing -> return ()
           Just b  -> do (v', ((p, c),l)) <- trainBatch v b
                         let ls' = take 20 (l:ls)
                             ps' = take 20 (p:ps)
                         liftIO.putStrLn$ showEFloat (Just 2) (median' ls')
                                        . showString " ("
                                        . showEFloat (Just 2) (minimum ls')
                                        . showString " .. "
                                        . showEFloat (Just 2) (maximum ls')
                                        . showString ")\t"

                                        . shows (median  ps')
                                        . showString " ("
                                        . shows (minimum ps')
                                        . showString " .. "
                                        . shows (maximum ps')
                                        . showString ")\t("

                                        . shows p
                                        . showString "/"
                                        . shows c
                                        . showString ", "
                                        . showEFloat (Just 4) l
                                        $ ")"
                         yield (Visor v')
                         go v' ls' ps'

batchify :: forall n a. (KnownNat n, Stack n (Widgets a)) => BatchC n a
batchify = do xs <- takeC n .| extractC .| sinkList
              case stack xs of
                Just xs' -> yield xs' >> batchify
                Nothing  -> return ()
  where
    n = fromInteger$ natVal (Proxy :: Proxy n)
    extractC = awaitForever$ \(shot, LabelVec ls) ->
      do xs <- Visor.extract shot
         yield (xs, ls)

saveMany :: Serialize a => String -> RTSink a
saveMany name = do liftIO$ createDirectoryIfMissing True dir'
                   go (0 :: Int)
  where dir' = dir </> name
        go i = do mx <- await
                  case mx of
                    Nothing -> return ()
                    Just x  -> do let path = dir' </> show i
                                  liftIO . putStrLn$ "Writing " ++ path
                                  liftIO$ BS.writeFile path (encode x)
                                  go (i+1)

loadMany :: Serialize a => String -> RTSource a
loadMany name = do sourceDirectory dir' .| awaitForever load
  where
    dir' = dir </> name
    load path = do bs <- liftIO$ BS.readFile path
                   let ex = decode bs
                   case ex of
                     Left err -> liftIO.putStrLn$ err
                     Right x  -> yield x

clear :: String -> IO ()
clear name = do createDirectoryIfMissing True (dir </> name)
                removeDirectoryRecursive$ dir </> name
                createDirectoryIfMissing True (dir </> name)

saveKernels :: (Head (NetConfig (Head (Widgets g))) ~ Convolution a 3 b c d e, GameState g)
            => Visor g -> IO ()
saveKernels (Visor v) = do clear "krns"
                           case v of
                             WNetwork (Convolution k _ _ `NCons` _) :- _ -> I.saveMany (dir</>"krns/") k
                             _ -> undefined

dir :: FilePath
dir = "data"
