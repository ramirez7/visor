module Volume.Internal where

import Label
import Data.Word
import Data.Array.Repa as R hiding ((++))
import qualified Data.Vector.Unboxed as DV
import Data.Array.Repa.Algorithms.Randomish
import GHC.Generics (Generic)
import Data.Serialize
import Volume.Types

{-# INLINE addConform #-}
-- extend a to b and add them
addConform :: Volume -> Volumes -> Array D DIM4 Double
addConform = undefined

-- Z:._:.a:.b:.c -> Z:.a:.b:.c
{-# INLINE sumOuter #-}
sumOuter :: Volumes -> Array D DIM3 Double
sumOuter = undefined

-- TODO: backprop van pooling moet extent-invariant worden
-- | Max-pooling function for volumes
pool :: Monad m => Volumes -> m Volumes
pool v = computeP $ R.traverse v shFn maxReg
  where
    n = 2
    shFn (Z:.n:.d:.h:.w) = Z:. n:.d :. h `div` n :. w `div` n
    maxReg lkUp (b:.y:.x) = maximum [ lkUp (b:.y*n + dy:.x*n + dx) | dy <- [0..n-1], dx <- [0 .. n-1]]

-- | Backprop of the max-pooling function. We upsample the error volume,
--   propagating the error to the position of the max element in every subregion,
--   setting the others to 0.
poolBackprop :: Monad m
             => Volumes -- ^ Input during forward pass, used to determine max-element
             -> Volumes -- ^ Output during forward pass, used to determine max-element
             -> Volumes -- ^ Error gradient on the output
             -> m Volumes -- ^ Error gradient on the input
poolBackprop input output errorGradient = computeP $ traverse3 input output errorGradient shFn outFn
  where
    n = 2
    shFn sh _ _ = sh
    {-# INLINE outFn #-}
    outFn in_ out_ err_ p@(b:.y:.x) = if out_ p' == in_ p then err_ p' else 0
      where p' = b :. y `div` n :. x `div` n

