{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}

module Lib where

import Types
import Network.Label
import Text.Read as T
import System.FilePath.Posix

divs :: Integral a => a -> a -> Bool
a `divs` b = b `mod` a == 0

count :: (a -> Bool) -> [a] -> Int
count p = length . filter p

splitDigits :: Int -> (Int, Int, Int)
splitDigits x = (d100, d10, d1)
  where (d100, r) = x `divMod` 100
        (d10, d1) = r `divMod` 10

toWidget :: Widget a => WLabel a -> a
toWidget (WLabel l) = case parseLabel l fromLabel of
                        Left str -> error str
                        Right a -> a

read' :: String -> Int
read' x = case T.readMaybe x of
           Just x -> x
           Nothing -> error$ "error parsing " ++ x

labelPath :: GameState g => Path g -> g
labelPath = delabel . parse

pmap :: (FilePath -> FilePath) -> Path a -> Path a
pmap f (Path p) = Path$ f p

showAndLabel :: GameState g => Path g -> String
showAndLabel p = show (pmap takeBaseName p) ++ "\t\t" ++ show (labelPath p)
