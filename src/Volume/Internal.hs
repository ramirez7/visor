{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE TypeOperators #-}

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
pool :: (Shape sh, Monad m) => ArrayU (sh:.Int:.Int) -> m (ArrayU (sh:.Int:.Int))
pool v = computeP $ R.traverse v shFn maxReg
  where
    n = 2
    shFn (b:.h:.w) = b :. h `div` n :. w `div` n
    maxReg lkUp (b:.y:.x) = maximum [ lkUp (b:.y*n + dy:.x*n + dx) | dy <- [0..n-1], dx <- [0 .. n-1]]

-- | Backprop of the max-pooling function. We upsample the error volume,
--   propagating the error to the position of the max element in every subregion,
--   setting the others to 0.
poolBackprop :: (Monad m, Shape sh)
             => ArrayU (sh:.Int:.Int) -- ^ Input during forward pass, used to determine max-element
             -> ArrayU (sh:.Int:.Int) -- ^ Output during forward pass, used to determine max-element
             -> ArrayU (sh:.Int:.Int) -- ^ Error gradient on the output
             -> m (ArrayU (sh:.Int:.Int)) -- ^ Error gradient on the input
poolBackprop input output errorGradient = computeP $ traverse3 input output errorGradient shFn outFn
  where
    n = 2
    shFn sh _ _ = sh
    {-# INLINE outFn #-}
    outFn in_ out_ err_ p@(b:.y:.x) = if out_ p' == in_ p then err_ p' else 0
      where p' = b :. y `div` n :. x `div` n

-- | Valid convolution of a stencil/kernel over some image, with the
--   kernel rotated 180 degrees. The valid part means that no zero
--   padding is applied. It is called corr to reflect that the
--   correct term for this operation would be cross-correlation.
--   Cross-correlation and convolution are used interchangeably in
--   most literature which can make things very confusing.
--   Note that the kernel is 4-dimensional, while the image is
--   three-dimensional. The stencils first dimension translates
--   to the output volume's depth.
--   A kernel of size (Z:. n_k :. d_k :. h_k :. w_k) convolved over
--   an image of size (Z:. d_i :. h_i :. w_i) results in
--   a output of size (Z:. n_k :. h_i - h_k +1 :. w_i - w_k + 1)
corr :: Monad m -- ^ Host monad for repa
     => Weights -- ^ Convolution kernel
     -> Volumes  -- ^ Image to iterate over
     -> m Volumes
corr krns img = if kd /= id
                   then error $ "kernel / image depth mismatch, k:" ++ show (extent krns) ++ " i:" ++ show (extent img)
                   else computeP $ fromFunction sh' convF
  where
    Z:.kn:.kd:.kh:.kw = extent krns
    Z:.si:.id:.ih:.iw = extent img
    sh' = Z:.si:.kn:.ih-kh+1:.iw-kw+1

    {-# INLINE convF #-}
    convF :: DIM4 -> Double
    convF (Z:.oi:.od:.oh:.ow) = sumAllS $ krn *^ reshape (extent krn) img'
      where
        krn  = slice krns (Z:.od:.All:.All:.All)
        img' = extract (Z:.oi:.0:.oh:.ow) (Z:.1:.id:.kh:.kw) img

corrVolumes :: (Monad m, Shape sh)
            => ArrayU (sh:.Int:.Int:.Int)
            -> ArrayU (sh:.Int:.Int:.Int)
            -> m Weights
corrVolumes krns imgs = computeP $ fromFunction sh' convF
  where
    kb:.kd:.kh:.kw = extent krns
    ib:.id:.ih:.iw = extent imgs
    sh' :: DIM4
    sh' = Z :. kd :. id :. ih-kh+1 :. iw-kw+1

    {-# INLINE convF #-}
    convF :: DIM4 -> Double
    convF (Z:.n:.z:.y:.x) = sumAllS $ krn *^ img
      where
        krn = extract (zeroDim :. n :. 0 :. 0) (kb :. 1 :. kh :. kw) krns
        img = extract (zeroDim :. z :. y :. x) (kb :. 1 :. kh :. kw) imgs
