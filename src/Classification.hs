module Classification where

-- | A label for some image classification. We make
--   explicit at the type level that classification may
--   fail. Classification fails when, for example, some feature
--   does not occur in some image. At the network level, Indeterminate
--   is represented as the ouput with index 0, which can lead to confusion
--   if we were to not use this explicit datatype.
data Label = Indeterminate | Label Int
  deriving (Eq, Show)


-- | Convert a label to the index used at the network level representation.
fromLabel :: Label -> Int
fromLabel Indeterminate = 0
fromLabel (Label n) = n + 1

-- | Convert an index used at the network level to a label
toLabel :: Int -> Label
toLabel n
  | n == 0    = Indeterminate
  | otherwise = Label (n-1)