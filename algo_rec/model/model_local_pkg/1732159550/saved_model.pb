╙▓
Г(ш'
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
╕
AsString

input"T

output"
Ttype:
2		
"
	precisionint         "

scientificbool( "
shortestbool( "
widthint         "
fillstring 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
S
	Bucketize

input"T

output"
Ttype:
2	"

boundarieslist(float)
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
н
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
k
NotEqual
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(Р
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
╖
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.52unknown╒М

global_step/Initializer/zerosConst*
_output_shapes
: *
value	B	 R *
_class
loc:@global_step*
dtype0	
П
global_step
VariableV2*
dtype0	*
	container *
_output_shapes
: *
shared_name *
shape: *
_class
loc:@global_step
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
validate_shape(*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
А
%ParseSingleExample/ParseSingleExamplePlaceholder*
dtype0	*
shape:         *#
_output_shapes
:         
В
'ParseSingleExample/ParseSingleExample_1Placeholder*#
_output_shapes
:         *
shape:         *
dtype0
В
'ParseSingleExample/ParseSingleExample_2Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
В
'ParseSingleExample/ParseSingleExample_3Placeholder*#
_output_shapes
:         *
shape:         *
dtype0
В
'ParseSingleExample/ParseSingleExample_4Placeholder*
dtype0*
shape:         *#
_output_shapes
:         
В
'ParseSingleExample/ParseSingleExample_5Placeholder*
shape:         *#
_output_shapes
:         *
dtype0	
В
'ParseSingleExample/ParseSingleExample_6Placeholder*
shape:         *#
_output_shapes
:         *
dtype0
В
'ParseSingleExample/ParseSingleExample_7Placeholder*
dtype0*
shape:         *#
_output_shapes
:         
В
'ParseSingleExample/ParseSingleExample_8Placeholder*#
_output_shapes
:         *
dtype0*
shape:         
В
'ParseSingleExample/ParseSingleExample_9Placeholder*#
_output_shapes
:         *
shape:         *
dtype0	
Г
(ParseSingleExample/ParseSingleExample_10Placeholder*
dtype0	*#
_output_shapes
:         *
shape:         
Г
(ParseSingleExample/ParseSingleExample_11Placeholder*#
_output_shapes
:         *
dtype0	*
shape:         
Г
(ParseSingleExample/ParseSingleExample_12Placeholder*
shape:         *
dtype0	*#
_output_shapes
:         
Г
(ParseSingleExample/ParseSingleExample_13Placeholder*
shape:         *
dtype0	*#
_output_shapes
:         
~
3input_layer/cate_level1_id_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
         *
dtype0
┘
/input_layer/cate_level1_id_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_13input_layer/cate_level1_id_embedding/ExpandDims/dim*

Tdim0*'
_output_shapes
:         *
T0
Д
Cinput_layer/cate_level1_id_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0
С
=input_layer/cate_level1_id_embedding/to_sparse_input/NotEqualNotEqual/input_layer/cate_level1_id_embedding/ExpandDimsCinput_layer/cate_level1_id_embedding/to_sparse_input/ignore_value/x*
incompatible_shape_error(*
T0*'
_output_shapes
:         
╢
<input_layer/cate_level1_id_embedding/to_sparse_input/indicesWhere=input_layer/cate_level1_id_embedding/to_sparse_input/NotEqual*
T0
*'
_output_shapes
:         
·
;input_layer/cate_level1_id_embedding/to_sparse_input/valuesGatherNd/input_layer/cate_level1_id_embedding/ExpandDims<input_layer/cate_level1_id_embedding/to_sparse_input/indices*#
_output_shapes
:         *
Tindices0	*
Tparams0
п
@input_layer/cate_level1_id_embedding/to_sparse_input/dense_shapeShape/input_layer/cate_level1_id_embedding/ExpandDims*
_output_shapes
:*
out_type0	*
T0
║
+input_layer/cate_level1_id_embedding/lookupStringToHashBucketFast;input_layer/cate_level1_id_embedding/to_sparse_input/values*#
_output_shapes
:         *
num_bucketsd
ї
Yinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
_output_shapes
:*
valueB"d      *
dtype0
ш
Xinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
dtype0*
_output_shapes
: *
valueB
 *    
ъ
Zinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
valueB
 *є╡>
ы
cinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/shape*
seed2 *

seed *
dtype0*
_output_shapes

:d*
T0*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights
У
Winput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/mulMulcinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalZinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes

:d*
T0*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights
Б
Sinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normalAddWinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/mulXinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
_output_shapes

:d
ї
6input_layer/cate_level1_id_embedding/embedding_weights
VariableV2*
shape
:d*
shared_name *I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
dtype0*
	container *
_output_shapes

:d
ё
=input_layer/cate_level1_id_embedding/embedding_weights/AssignAssign6input_layer/cate_level1_id_embedding/embedding_weightsSinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal*
T0*
_output_shapes

:d*
validate_shape(*
use_locking(*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights
є
;input_layer/cate_level1_id_embedding/embedding_weights/readIdentity6input_layer/cate_level1_id_embedding/embedding_weights*
_output_shapes

:d*
T0*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights
Ы
Qinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ъ
Pinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
э
Kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SliceSlice@input_layer/cate_level1_id_embedding/to_sparse_input/dense_shapeQinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice/beginPinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice/size*
_output_shapes
:*
T0	*
Index0
Х
Kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
к
Jinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ProdProdKinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SliceKinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Const*
	keep_dims( *
T0	*

Tidx0*
_output_shapes
: 
Ш
Vinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Х
Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Я
Ninput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2GatherV2@input_layer/cate_level1_id_embedding/to_sparse_input/dense_shapeVinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2/indicesSinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2/axis*
Tparams0	*

batch_dims *
Tindices0*
_output_shapes
: *
Taxis0
к
Linput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Cast/xPackJinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ProdNinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2*

axis *
_output_shapes
:*
N*
T0	
с
Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseReshapeSparseReshape<input_layer/cate_level1_id_embedding/to_sparse_input/indices@input_layer/cate_level1_id_embedding/to_sparse_input/dense_shapeLinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Cast/x*-
_output_shapes
:         :
├
\input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseReshape/IdentityIdentity+input_layer/cate_level1_id_embedding/lookup*
T0	*#
_output_shapes
:         
Ц
Tinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
─
Rinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GreaterEqualGreaterEqual\input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseReshape/IdentityTinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GreaterEqual/y*#
_output_shapes
:         *
T0	
┌
Kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/WhereWhereRinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GreaterEqual*
T0
*'
_output_shapes
:         
ж
Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
╢
Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ReshapeReshapeKinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/WhereSinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0	
Ч
Uinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╛
Pinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_1GatherV2Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseReshapeMinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ReshapeUinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*'
_output_shapes
:         *

batch_dims *
Tparams0	
Ч
Uinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
├
Pinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_2GatherV2\input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseReshape/IdentityMinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ReshapeUinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_2/axis*

batch_dims *
Tparams0	*
Taxis0*
Tindices0	*#
_output_shapes
:         
╓
Ninput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/IdentityIdentityUinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
б
_input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
╕
minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_1Pinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/GatherV2_2Ninput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Identity_input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
┬
qinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
─
sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
─
sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
ц
kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceminput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsqinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice/stacksinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
end_mask*

begin_mask*#
_output_shapes
:         *
Index0*
ellipsis_mask *
T0	
д
binput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/CastCastkinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*
Truncate( *#
_output_shapes
:         *

SrcT0	
л
dinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueoinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
out_idx0*
T0	*2
_output_shapes 
:         :         
А
sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
dtype0*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
value	B : 
─
ninput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2;input_layer/cate_level1_id_embedding/embedding_weights/readdinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/Uniquesinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*
Tindices0	*
Taxis0*

batch_dims *'
_output_shapes
:         *I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights
е
winput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityninput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
ї
]input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityfinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/Unique:1binput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:         *

Tidx0*
T0
ж
Uinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       
т
Oinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_1Reshapeoinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Uinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_1/shape*
Tshape0*
T0
*'
_output_shapes
:         
ш
Kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ShapeShape]input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
г
Yinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
е
[input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
е
[input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╫
Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_sliceStridedSliceKinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/ShapeYinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice/stack[input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice/stack_1[input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice/stack_2*
_output_shapes
: *

begin_mask *
Index0*
ellipsis_mask *
shrink_axis_mask*
T0*
end_mask *
new_axis_mask 
П
Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
▒
Kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/stackPackMinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/stack/0Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N*

axis 
╜
Jinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/TileTileOinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_1Kinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
ю
Pinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/zeros_like	ZerosLike]input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
■
Einput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weightsSelectJinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/TilePinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/zeros_like]input_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┌
Linput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Cast_1Cast@input_layer/cate_level1_id_embedding/to_sparse_input/dense_shape*

DstT0*
_output_shapes
:*

SrcT0	*
Truncate( 
Э
Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ь
Rinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
 
Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_1SliceLinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Cast_1Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_1/beginRinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_1/size*
T0*
_output_shapes
:*
Index0
╥
Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Shape_1ShapeEinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights*
T0*
out_type0*
_output_shapes
:
Э
Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
е
Rinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
         
А
Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_2SliceMinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Shape_1Sinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_2/beginRinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_2/size*
T0*
_output_shapes
:*
Index0
У
Qinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
Г
Linput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/concatConcatV2Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_1Minput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Slice_2Qinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
п
Oinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_2ReshapeEinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weightsLinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/concat*'
_output_shapes
:         *
Tshape0*
T0
╣
*input_layer/cate_level1_id_embedding/ShapeShapeOinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
В
8input_layer/cate_level1_id_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Д
:input_layer/cate_level1_id_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Д
:input_layer/cate_level1_id_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
▓
2input_layer/cate_level1_id_embedding/strided_sliceStridedSlice*input_layer/cate_level1_id_embedding/Shape8input_layer/cate_level1_id_embedding/strided_slice/stack:input_layer/cate_level1_id_embedding/strided_slice/stack_1:input_layer/cate_level1_id_embedding/strided_slice/stack_2*

begin_mask *
_output_shapes
: *
ellipsis_mask *
Index0*
new_axis_mask *
shrink_axis_mask*
end_mask *
T0
v
4input_layer/cate_level1_id_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
▐
2input_layer/cate_level1_id_embedding/Reshape/shapePack2input_layer/cate_level1_id_embedding/strided_slice4input_layer/cate_level1_id_embedding/Reshape/shape/1*
_output_shapes
:*
T0*
N*

axis 
№
,input_layer/cate_level1_id_embedding/ReshapeReshapeOinput_layer/cate_level1_id_embedding/cate_level1_id_embedding_weights/Reshape_22input_layer/cate_level1_id_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
~
3input_layer/cate_level2_id_embedding/ExpandDims/dimConst*
valueB :
         *
_output_shapes
: *
dtype0
┘
/input_layer/cate_level2_id_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_23input_layer/cate_level2_id_embedding/ExpandDims/dim*'
_output_shapes
:         *
T0*

Tdim0
Д
Cinput_layer/cate_level2_id_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
С
=input_layer/cate_level2_id_embedding/to_sparse_input/NotEqualNotEqual/input_layer/cate_level2_id_embedding/ExpandDimsCinput_layer/cate_level2_id_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:         *
incompatible_shape_error(
╢
<input_layer/cate_level2_id_embedding/to_sparse_input/indicesWhere=input_layer/cate_level2_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         *
T0

·
;input_layer/cate_level2_id_embedding/to_sparse_input/valuesGatherNd/input_layer/cate_level2_id_embedding/ExpandDims<input_layer/cate_level2_id_embedding/to_sparse_input/indices*
Tparams0*
Tindices0	*#
_output_shapes
:         
п
@input_layer/cate_level2_id_embedding/to_sparse_input/dense_shapeShape/input_layer/cate_level2_id_embedding/ExpandDims*
T0*
_output_shapes
:*
out_type0	
╗
+input_layer/cate_level2_id_embedding/lookupStringToHashBucketFast;input_layer/cate_level2_id_embedding/to_sparse_input/values*
num_bucketsР*#
_output_shapes
:         
ї
Yinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"Р     *
dtype0*
_output_shapes
:*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights
ш
Xinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    
ъ
Zinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *є╡>*
_output_shapes
: *
dtype0*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights
ь
cinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/shape*
dtype0*
T0*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights*
seed2 *
_output_shapes
:	Р*

seed 
Ф
Winput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/mulMulcinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalZinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/stddev*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights*
_output_shapes
:	Р*
T0
В
Sinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normalAddWinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/mulXinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	Р*
T0*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights
ў
6input_layer/cate_level2_id_embedding/embedding_weights
VariableV2*
shape:	Р*
_output_shapes
:	Р*
dtype0*
shared_name *
	container *I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights
Є
=input_layer/cate_level2_id_embedding/embedding_weights/AssignAssign6input_layer/cate_level2_id_embedding/embedding_weightsSinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal*
use_locking(*
_output_shapes
:	Р*
T0*
validate_shape(*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights
Ї
;input_layer/cate_level2_id_embedding/embedding_weights/readIdentity6input_layer/cate_level2_id_embedding/embedding_weights*
T0*
_output_shapes
:	Р*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights
Ы
Qinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
Ъ
Pinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
э
Kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SliceSlice@input_layer/cate_level2_id_embedding/to_sparse_input/dense_shapeQinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice/beginPinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Х
Kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
к
Jinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ProdProdKinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SliceKinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Const*

Tidx0*
T0	*
	keep_dims( *
_output_shapes
: 
Ш
Vinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2/indicesConst*
dtype0*
value	B :*
_output_shapes
: 
Х
Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Я
Ninput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2GatherV2@input_layer/cate_level2_id_embedding/to_sparse_input/dense_shapeVinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2/indicesSinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2/axis*
Tparams0	*
Tindices0*

batch_dims *
_output_shapes
: *
Taxis0
к
Linput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Cast/xPackJinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ProdNinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2*

axis *
_output_shapes
:*
N*
T0	
с
Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseReshapeSparseReshape<input_layer/cate_level2_id_embedding/to_sparse_input/indices@input_layer/cate_level2_id_embedding/to_sparse_input/dense_shapeLinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Cast/x*-
_output_shapes
:         :
├
\input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseReshape/IdentityIdentity+input_layer/cate_level2_id_embedding/lookup*#
_output_shapes
:         *
T0	
Ц
Tinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GreaterEqual/yConst*
dtype0	*
value	B	 R *
_output_shapes
: 
─
Rinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GreaterEqualGreaterEqual\input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseReshape/IdentityTinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
┌
Kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/WhereWhereRinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

ж
Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
╢
Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ReshapeReshapeKinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/WhereSinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:         *
Tshape0
Ч
Uinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╛
Pinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_1GatherV2Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseReshapeMinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ReshapeUinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:         *
Taxis0*

batch_dims 
Ч
Uinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
├
Pinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_2GatherV2\input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseReshape/IdentityMinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ReshapeUinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_2/axis*

batch_dims *
Tparams0	*
Taxis0*
Tindices0	*#
_output_shapes
:         
╓
Ninput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/IdentityIdentityUinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
б
_input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
╕
minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_1Pinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/GatherV2_2Ninput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Identity_input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
┬
qinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
─
sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
─
sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
ц
kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceminput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsqinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice/stacksinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
Index0*#
_output_shapes
:         *
end_mask*
T0	*
shrink_axis_mask*
new_axis_mask 
д
binput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/CastCastkinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:         *

DstT0*

SrcT0	*
Truncate( 
л
dinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueoinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:         :         *
out_idx0
А
sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
_output_shapes
: *I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights*
dtype0
─
ninput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2;input_layer/cate_level2_id_embedding/embedding_weights/readdinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/Uniquesinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*

batch_dims *
Tindices0	*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights*
Taxis0*'
_output_shapes
:         *
Tparams0
е
winput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityninput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
ї
]input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityfinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/Unique:1binput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*'
_output_shapes
:         *
T0
ж
Uinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
т
Oinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_1Reshapeoinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Uinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_1/shape*
Tshape0*'
_output_shapes
:         *
T0

ш
Kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ShapeShape]input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
г
Yinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
е
[input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
е
[input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╫
Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_sliceStridedSliceKinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/ShapeYinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice/stack[input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice/stack_1[input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice/stack_2*

begin_mask *
end_mask *
_output_shapes
: *
new_axis_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask 
П
Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
▒
Kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/stackPackMinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/stack/0Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/strided_slice*
_output_shapes
:*
N*

axis *
T0
╜
Jinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/TileTileOinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_1Kinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
ю
Pinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/zeros_like	ZerosLike]input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
■
Einput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weightsSelectJinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/TilePinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/zeros_like]input_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┌
Linput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Cast_1Cast@input_layer/cate_level2_id_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	*
Truncate( 
Э
Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
Ь
Rinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
 
Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_1SliceLinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Cast_1Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_1/beginRinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
╥
Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Shape_1ShapeEinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights*
_output_shapes
:*
T0*
out_type0
Э
Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
е
Rinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
А
Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_2SliceMinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Shape_1Sinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_2/beginRinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_2/size*
_output_shapes
:*
T0*
Index0
У
Qinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Г
Linput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/concatConcatV2Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_1Minput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Slice_2Qinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
п
Oinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_2ReshapeEinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weightsLinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:         
╣
*input_layer/cate_level2_id_embedding/ShapeShapeOinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_2*
out_type0*
T0*
_output_shapes
:
В
8input_layer/cate_level2_id_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Д
:input_layer/cate_level2_id_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Д
:input_layer/cate_level2_id_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
▓
2input_layer/cate_level2_id_embedding/strided_sliceStridedSlice*input_layer/cate_level2_id_embedding/Shape8input_layer/cate_level2_id_embedding/strided_slice/stack:input_layer/cate_level2_id_embedding/strided_slice/stack_1:input_layer/cate_level2_id_embedding/strided_slice/stack_2*
end_mask *
Index0*
_output_shapes
: *
new_axis_mask *

begin_mask *
ellipsis_mask *
T0*
shrink_axis_mask
v
4input_layer/cate_level2_id_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
▐
2input_layer/cate_level2_id_embedding/Reshape/shapePack2input_layer/cate_level2_id_embedding/strided_slice4input_layer/cate_level2_id_embedding/Reshape/shape/1*
T0*
_output_shapes
:*
N*

axis 
№
,input_layer/cate_level2_id_embedding/ReshapeReshapeOinput_layer/cate_level2_id_embedding/cate_level2_id_embedding_weights/Reshape_22input_layer/cate_level2_id_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
~
3input_layer/cate_level3_id_embedding/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
┘
/input_layer/cate_level3_id_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_33input_layer/cate_level3_id_embedding/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
Д
Cinput_layer/cate_level3_id_embedding/to_sparse_input/ignore_value/xConst*
dtype0*
valueB B *
_output_shapes
: 
С
=input_layer/cate_level3_id_embedding/to_sparse_input/NotEqualNotEqual/input_layer/cate_level3_id_embedding/ExpandDimsCinput_layer/cate_level3_id_embedding/to_sparse_input/ignore_value/x*
T0*
incompatible_shape_error(*'
_output_shapes
:         
╢
<input_layer/cate_level3_id_embedding/to_sparse_input/indicesWhere=input_layer/cate_level3_id_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         *
T0

·
;input_layer/cate_level3_id_embedding/to_sparse_input/valuesGatherNd/input_layer/cate_level3_id_embedding/ExpandDims<input_layer/cate_level3_id_embedding/to_sparse_input/indices*#
_output_shapes
:         *
Tparams0*
Tindices0	
п
@input_layer/cate_level3_id_embedding/to_sparse_input/dense_shapeShape/input_layer/cate_level3_id_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
╗
+input_layer/cate_level3_id_embedding/lookupStringToHashBucketFast;input_layer/cate_level3_id_embedding/to_sparse_input/values*#
_output_shapes
:         *
num_bucketsш
ї
Yinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"ш     *I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
_output_shapes
:
ш
Xinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights
ъ
Zinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
valueB
 *є╡>*
dtype0
ь
cinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/shape*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*

seed *
dtype0*
seed2 *
_output_shapes
:	ш*
T0
Ф
Winput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/mulMulcinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalZinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/stddev*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
_output_shapes
:	ш*
T0
В
Sinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normalAddWinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/mulXinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	ш*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
T0
ў
6input_layer/cate_level3_id_embedding/embedding_weights
VariableV2*
shared_name *
shape:	ш*
	container *
_output_shapes
:	ш*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
dtype0
Є
=input_layer/cate_level3_id_embedding/embedding_weights/AssignAssign6input_layer/cate_level3_id_embedding/embedding_weightsSinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal*
use_locking(*
validate_shape(*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
_output_shapes
:	ш*
T0
Ї
;input_layer/cate_level3_id_embedding/embedding_weights/readIdentity6input_layer/cate_level3_id_embedding/embedding_weights*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
_output_shapes
:	ш*
T0
Ы
Qinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ъ
Pinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
э
Kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SliceSlice@input_layer/cate_level3_id_embedding/to_sparse_input/dense_shapeQinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice/beginPinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
Х
Kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ConstConst*
valueB: *
_output_shapes
:*
dtype0
к
Jinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ProdProdKinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SliceKinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Const*
_output_shapes
: *
	keep_dims( *
T0	*

Tidx0
Ш
Vinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Х
Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Я
Ninput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2GatherV2@input_layer/cate_level3_id_embedding/to_sparse_input/dense_shapeVinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2/indicesSinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2/axis*
Taxis0*
_output_shapes
: *
Tindices0*

batch_dims *
Tparams0	
к
Linput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Cast/xPackJinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ProdNinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2*
T0	*
_output_shapes
:*

axis *
N
с
Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseReshapeSparseReshape<input_layer/cate_level3_id_embedding/to_sparse_input/indices@input_layer/cate_level3_id_embedding/to_sparse_input/dense_shapeLinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Cast/x*-
_output_shapes
:         :
├
\input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseReshape/IdentityIdentity+input_layer/cate_level3_id_embedding/lookup*#
_output_shapes
:         *
T0	
Ц
Tinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
─
Rinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GreaterEqualGreaterEqual\input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseReshape/IdentityTinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
┌
Kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/WhereWhereRinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GreaterEqual*
T0
*'
_output_shapes
:         
ж
Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╢
Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ReshapeReshapeKinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/WhereSinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:         
Ч
Uinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
╛
Pinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_1GatherV2Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseReshapeMinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ReshapeUinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_1/axis*

batch_dims *
Taxis0*'
_output_shapes
:         *
Tindices0	*
Tparams0	
Ч
Uinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
├
Pinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_2GatherV2\input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseReshape/IdentityMinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ReshapeUinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_2/axis*

batch_dims *
Tindices0	*
Taxis0*#
_output_shapes
:         *
Tparams0	
╓
Ninput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/IdentityIdentityUinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
б
_input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
╕
minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_1Pinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/GatherV2_2Ninput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Identity_input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
┬
qinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
─
sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
─
sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ц
kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceminput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsqinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice/stacksinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
ellipsis_mask *
end_mask*

begin_mask*
shrink_axis_mask*
Index0*
new_axis_mask *#
_output_shapes
:         
д
binput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/CastCastkinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/strided_slice*
Truncate( *#
_output_shapes
:         *

DstT0*

SrcT0	
л
dinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueoinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
out_idx0*2
_output_shapes 
:         :         *
T0	
А
sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
dtype0*
value	B : *I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights
─
ninput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2;input_layer/cate_level3_id_embedding/embedding_weights/readdinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/Uniquesinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*

batch_dims *I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
Tindices0	*
Tparams0*'
_output_shapes
:         
е
winput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityninput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
ї
]input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityfinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/Unique:1binput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*'
_output_shapes
:         *
T0
ж
Uinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
т
Oinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_1Reshapeoinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Uinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:         *
Tshape0
ш
Kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ShapeShape]input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse*
out_type0*
_output_shapes
:*
T0
г
Yinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
е
[input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
е
[input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╫
Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_sliceStridedSliceKinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/ShapeYinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice/stack[input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice/stack_1[input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice/stack_2*

begin_mask *
new_axis_mask *
T0*
ellipsis_mask *
shrink_axis_mask*
Index0*
end_mask *
_output_shapes
: 
П
Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
▒
Kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/stackPackMinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/stack/0Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/strided_slice*
T0*

axis *
N*
_output_shapes
:
╜
Jinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/TileTileOinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_1Kinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/stack*0
_output_shapes
:                  *

Tmultiples0*
T0

ю
Pinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/zeros_like	ZerosLike]input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
■
Einput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weightsSelectJinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/TilePinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/zeros_like]input_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┌
Linput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Cast_1Cast@input_layer/cate_level3_id_embedding/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	*
Truncate( 
Э
Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
Ь
Rinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
 
Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_1SliceLinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Cast_1Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_1/beginRinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
╥
Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Shape_1ShapeEinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights*
out_type0*
_output_shapes
:*
T0
Э
Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
е
Rinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         
А
Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_2SliceMinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Shape_1Sinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_2/beginRinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
У
Qinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Г
Linput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/concatConcatV2Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_1Minput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Slice_2Qinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
п
Oinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_2ReshapeEinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weightsLinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/concat*
T0*'
_output_shapes
:         *
Tshape0
╣
*input_layer/cate_level3_id_embedding/ShapeShapeOinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_2*
T0*
_output_shapes
:*
out_type0
В
8input_layer/cate_level3_id_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
Д
:input_layer/cate_level3_id_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Д
:input_layer/cate_level3_id_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
▓
2input_layer/cate_level3_id_embedding/strided_sliceStridedSlice*input_layer/cate_level3_id_embedding/Shape8input_layer/cate_level3_id_embedding/strided_slice/stack:input_layer/cate_level3_id_embedding/strided_slice/stack_1:input_layer/cate_level3_id_embedding/strided_slice/stack_2*
end_mask *
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
_output_shapes
: *
T0*
Index0
v
4input_layer/cate_level3_id_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
▐
2input_layer/cate_level3_id_embedding/Reshape/shapePack2input_layer/cate_level3_id_embedding/strided_slice4input_layer/cate_level3_id_embedding/Reshape/shape/1*
_output_shapes
:*

axis *
N*
T0
№
,input_layer/cate_level3_id_embedding/ReshapeReshapeOinput_layer/cate_level3_id_embedding/cate_level3_id_embedding_weights/Reshape_22input_layer/cate_level3_id_embedding/Reshape/shape*'
_output_shapes
:         *
Tshape0*
T0
~
3input_layer/cate_level4_id_embedding/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
┘
/input_layer/cate_level4_id_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_43input_layer/cate_level4_id_embedding/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:         
Д
Cinput_layer/cate_level4_id_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
_output_shapes
: *
dtype0
С
=input_layer/cate_level4_id_embedding/to_sparse_input/NotEqualNotEqual/input_layer/cate_level4_id_embedding/ExpandDimsCinput_layer/cate_level4_id_embedding/to_sparse_input/ignore_value/x*
incompatible_shape_error(*
T0*'
_output_shapes
:         
╢
<input_layer/cate_level4_id_embedding/to_sparse_input/indicesWhere=input_layer/cate_level4_id_embedding/to_sparse_input/NotEqual*
T0
*'
_output_shapes
:         
·
;input_layer/cate_level4_id_embedding/to_sparse_input/valuesGatherNd/input_layer/cate_level4_id_embedding/ExpandDims<input_layer/cate_level4_id_embedding/to_sparse_input/indices*
Tindices0	*#
_output_shapes
:         *
Tparams0
п
@input_layer/cate_level4_id_embedding/to_sparse_input/dense_shapeShape/input_layer/cate_level4_id_embedding/ExpandDims*
out_type0	*
_output_shapes
:*
T0
╗
+input_layer/cate_level4_id_embedding/lookupStringToHashBucketFast;input_layer/cate_level4_id_embedding/to_sparse_input/values*#
_output_shapes
:         *
num_buckets╨
ї
Yinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"╨     *I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights
ш
Xinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    
ъ
Zinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
valueB
 *є╡>
ь
cinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/shape*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
seed2 *

seed *
_output_shapes
:	╨*
dtype0*
T0
Ф
Winput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/mulMulcinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalZinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
_output_shapes
:	╨
В
Sinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normalAddWinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/mulXinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes
:	╨*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
T0
ў
6input_layer/cate_level4_id_embedding/embedding_weights
VariableV2*
_output_shapes
:	╨*
shape:	╨*
	container *I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
dtype0*
shared_name 
Є
=input_layer/cate_level4_id_embedding/embedding_weights/AssignAssign6input_layer/cate_level4_id_embedding/embedding_weightsSinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes
:	╨*
T0*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
use_locking(*
validate_shape(
Ї
;input_layer/cate_level4_id_embedding/embedding_weights/readIdentity6input_layer/cate_level4_id_embedding/embedding_weights*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
T0*
_output_shapes
:	╨
Ы
Qinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ъ
Pinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
э
Kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SliceSlice@input_layer/cate_level4_id_embedding/to_sparse_input/dense_shapeQinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice/beginPinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
Х
Kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
к
Jinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ProdProdKinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SliceKinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Const*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0	
Ш
Vinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
Х
Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
Я
Ninput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2GatherV2@input_layer/cate_level4_id_embedding/to_sparse_input/dense_shapeVinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2/indicesSinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2/axis*

batch_dims *
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
к
Linput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Cast/xPackJinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ProdNinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2*
_output_shapes
:*

axis *
N*
T0	
с
Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseReshapeSparseReshape<input_layer/cate_level4_id_embedding/to_sparse_input/indices@input_layer/cate_level4_id_embedding/to_sparse_input/dense_shapeLinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Cast/x*-
_output_shapes
:         :
├
\input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseReshape/IdentityIdentity+input_layer/cate_level4_id_embedding/lookup*#
_output_shapes
:         *
T0	
Ц
Tinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
─
Rinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GreaterEqualGreaterEqual\input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseReshape/IdentityTinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
┌
Kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/WhereWhereRinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

ж
Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╢
Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ReshapeReshapeKinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/WhereSinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0	
Ч
Uinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╛
Pinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_1GatherV2Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseReshapeMinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ReshapeUinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:         *
Tindices0	*

batch_dims *
Taxis0
Ч
Uinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
├
Pinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_2GatherV2\input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseReshape/IdentityMinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ReshapeUinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_2/axis*
Taxis0*

batch_dims *
Tparams0	*#
_output_shapes
:         *
Tindices0	
╓
Ninput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/IdentityIdentityUinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
б
_input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 
╕
minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_1Pinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/GatherV2_2Ninput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Identity_input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0	
┬
qinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
─
sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
_output_shapes
:*
dtype0
─
sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
ц
kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceminput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsqinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice/stacksinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *
end_mask*
shrink_axis_mask*
T0	*#
_output_shapes
:         *
Index0*
ellipsis_mask *

begin_mask
д
binput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/CastCastkinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:         *

DstT0*
Truncate( 
л
dinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/UniqueUniqueoinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:         :         *
out_idx0*
T0	
А
sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
dtype0*
_output_shapes
: 
─
ninput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2;input_layer/cate_level4_id_embedding/embedding_weights/readdinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/Uniquesinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*'
_output_shapes
:         *
Taxis0*

batch_dims *I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
Tindices0	
е
winput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityninput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:         
ї
]input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparseSparseSegmentMeanwinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityfinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/Unique:1binput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:         *

Tidx0*
T0
ж
Uinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_1/shapeConst*
valueB"       *
_output_shapes
:*
dtype0
т
Oinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_1Reshapeoinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Uinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_1/shape*
Tshape0*'
_output_shapes
:         *
T0

ш
Kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ShapeShape]input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
г
Yinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
е
[input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
е
[input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╫
Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_sliceStridedSliceKinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/ShapeYinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice/stack[input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice/stack_1[input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
ellipsis_mask *
shrink_axis_mask*

begin_mask *
end_mask *
T0*
Index0*
new_axis_mask 
П
Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
▒
Kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/stackPackMinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/stack/0Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/strided_slice*
_output_shapes
:*
N*
T0*

axis 
╜
Jinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/TileTileOinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_1Kinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
ю
Pinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/zeros_like	ZerosLike]input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
■
Einput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weightsSelectJinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/TilePinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/zeros_like]input_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┌
Linput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Cast_1Cast@input_layer/cate_level4_id_embedding/to_sparse_input/dense_shape*

DstT0*
Truncate( *

SrcT0	*
_output_shapes
:
Э
Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
Ь
Rinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
 
Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_1SliceLinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Cast_1Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_1/beginRinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
╥
Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Shape_1ShapeEinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights*
_output_shapes
:*
out_type0*
T0
Э
Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
е
Rinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         
А
Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_2SliceMinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Shape_1Sinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_2/beginRinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
У
Qinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Г
Linput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/concatConcatV2Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_1Minput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Slice_2Qinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/concat/axis*

Tidx0*
_output_shapes
:*
T0*
N
п
Oinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_2ReshapeEinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weightsLinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/concat*'
_output_shapes
:         *
Tshape0*
T0
╣
*input_layer/cate_level4_id_embedding/ShapeShapeOinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
В
8input_layer/cate_level4_id_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Д
:input_layer/cate_level4_id_embedding/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Д
:input_layer/cate_level4_id_embedding/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
▓
2input_layer/cate_level4_id_embedding/strided_sliceStridedSlice*input_layer/cate_level4_id_embedding/Shape8input_layer/cate_level4_id_embedding/strided_slice/stack:input_layer/cate_level4_id_embedding/strided_slice/stack_1:input_layer/cate_level4_id_embedding/strided_slice/stack_2*
Index0*
shrink_axis_mask*
ellipsis_mask *
end_mask *

begin_mask *
T0*
_output_shapes
: *
new_axis_mask 
v
4input_layer/cate_level4_id_embedding/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
▐
2input_layer/cate_level4_id_embedding/Reshape/shapePack2input_layer/cate_level4_id_embedding/strided_slice4input_layer/cate_level4_id_embedding/Reshape/shape/1*
T0*

axis *
_output_shapes
:*
N
№
,input_layer/cate_level4_id_embedding/ReshapeReshapeOinput_layer/cate_level4_id_embedding/cate_level4_id_embedding_weights/Reshape_22input_layer/cate_level4_id_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
w
,input_layer/country_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
         *
dtype0
╦
(input_layer/country_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_6,input_layer/country_embedding/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
}
<input_layer/country_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
№
6input_layer/country_embedding/to_sparse_input/NotEqualNotEqual(input_layer/country_embedding/ExpandDims<input_layer/country_embedding/to_sparse_input/ignore_value/x*
T0*
incompatible_shape_error(*'
_output_shapes
:         
и
5input_layer/country_embedding/to_sparse_input/indicesWhere6input_layer/country_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         *
T0

х
4input_layer/country_embedding/to_sparse_input/valuesGatherNd(input_layer/country_embedding/ExpandDims5input_layer/country_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:         
б
9input_layer/country_embedding/to_sparse_input/dense_shapeShape(input_layer/country_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
м
$input_layer/country_embedding/lookupStringToHashBucketFast4input_layer/country_embedding/to_sparse_input/values*
num_buckets*#
_output_shapes
:         
ч
Rinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *
dtype0*B
_class8
64loc:@input_layer/country_embedding/embedding_weights
┌
Qinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
_output_shapes
: 
▄
Sinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
valueB
 *   ?*
dtype0*
_output_shapes
: 
╓
\input_layer/country_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/shape*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
T0*
dtype0*

seed *
_output_shapes

:*
seed2 
ў
Pinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/mulMul\input_layer/country_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalSinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
_output_shapes

:
х
Linput_layer/country_embedding/embedding_weights/Initializer/truncated_normalAddPinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/mulQinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal/mean*
_output_shapes

:*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
T0
ч
/input_layer/country_embedding/embedding_weights
VariableV2*
	container *
dtype0*
_output_shapes

:*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
shared_name *
shape
:
╒
6input_layer/country_embedding/embedding_weights/AssignAssign/input_layer/country_embedding/embedding_weightsLinput_layer/country_embedding/embedding_weights/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*B
_class8
64loc:@input_layer/country_embedding/embedding_weights
▐
4input_layer/country_embedding/embedding_weights/readIdentity/input_layer/country_embedding/embedding_weights*
_output_shapes

:*
T0*B
_class8
64loc:@input_layer/country_embedding/embedding_weights
Н
Cinput_layer/country_embedding/country_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
М
Binput_layer/country_embedding/country_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
╝
=input_layer/country_embedding/country_embedding_weights/SliceSlice9input_layer/country_embedding/to_sparse_input/dense_shapeCinput_layer/country_embedding/country_embedding_weights/Slice/beginBinput_layer/country_embedding/country_embedding_weights/Slice/size*
T0	*
_output_shapes
:*
Index0
З
=input_layer/country_embedding/country_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
А
<input_layer/country_embedding/country_embedding_weights/ProdProd=input_layer/country_embedding/country_embedding_weights/Slice=input_layer/country_embedding/country_embedding_weights/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
К
Hinput_layer/country_embedding/country_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
З
Einput_layer/country_embedding/country_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
ю
@input_layer/country_embedding/country_embedding_weights/GatherV2GatherV29input_layer/country_embedding/to_sparse_input/dense_shapeHinput_layer/country_embedding/country_embedding_weights/GatherV2/indicesEinput_layer/country_embedding/country_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*

batch_dims *
_output_shapes
: *
Tparams0	
А
>input_layer/country_embedding/country_embedding_weights/Cast/xPack<input_layer/country_embedding/country_embedding_weights/Prod@input_layer/country_embedding/country_embedding_weights/GatherV2*
N*
_output_shapes
:*

axis *
T0	
╖
Einput_layer/country_embedding/country_embedding_weights/SparseReshapeSparseReshape5input_layer/country_embedding/to_sparse_input/indices9input_layer/country_embedding/to_sparse_input/dense_shape>input_layer/country_embedding/country_embedding_weights/Cast/x*-
_output_shapes
:         :
о
Ninput_layer/country_embedding/country_embedding_weights/SparseReshape/IdentityIdentity$input_layer/country_embedding/lookup*
T0	*#
_output_shapes
:         
И
Finput_layer/country_embedding/country_embedding_weights/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ъ
Dinput_layer/country_embedding/country_embedding_weights/GreaterEqualGreaterEqualNinput_layer/country_embedding/country_embedding_weights/SparseReshape/IdentityFinput_layer/country_embedding/country_embedding_weights/GreaterEqual/y*#
_output_shapes
:         *
T0	
╛
=input_layer/country_embedding/country_embedding_weights/WhereWhereDinput_layer/country_embedding/country_embedding_weights/GreaterEqual*
T0
*'
_output_shapes
:         
Ш
Einput_layer/country_embedding/country_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
М
?input_layer/country_embedding/country_embedding_weights/ReshapeReshape=input_layer/country_embedding/country_embedding_weights/WhereEinput_layer/country_embedding/country_embedding_weights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
Й
Ginput_layer/country_embedding/country_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ж
Binput_layer/country_embedding/country_embedding_weights/GatherV2_1GatherV2Einput_layer/country_embedding/country_embedding_weights/SparseReshape?input_layer/country_embedding/country_embedding_weights/ReshapeGinput_layer/country_embedding/country_embedding_weights/GatherV2_1/axis*
Taxis0*
Tparams0	*'
_output_shapes
:         *

batch_dims *
Tindices0	
Й
Ginput_layer/country_embedding/country_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Л
Binput_layer/country_embedding/country_embedding_weights/GatherV2_2GatherV2Ninput_layer/country_embedding/country_embedding_weights/SparseReshape/Identity?input_layer/country_embedding/country_embedding_weights/ReshapeGinput_layer/country_embedding/country_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*#
_output_shapes
:         *

batch_dims *
Tparams0	
║
@input_layer/country_embedding/country_embedding_weights/IdentityIdentityGinput_layer/country_embedding/country_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
У
Qinput_layer/country_embedding/country_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B	 R *
dtype0	
Є
_input_layer/country_embedding/country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsBinput_layer/country_embedding/country_embedding_weights/GatherV2_1Binput_layer/country_embedding/country_embedding_weights/GatherV2_2@input_layer/country_embedding/country_embedding_weights/IdentityQinput_layer/country_embedding/country_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
┤
cinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
╢
einput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
╢
einput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
а
]input_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice_input_layer/country_embedding/country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowscinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice/stackeinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1einput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
shrink_axis_mask*
Index0*#
_output_shapes
:         *
end_mask*
ellipsis_mask *
T0	*
new_axis_mask 
И
Tinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/CastCast]input_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:         *
Truncate( *

SrcT0	*

DstT0
П
Vinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/UniqueUniqueainput_layer/country_embedding/country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:         :         *
T0	*
out_idx0
ы
einput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
dtype0*
value	B : *B
_class8
64loc:@input_layer/country_embedding/embedding_weights
М
`input_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV24input_layer/country_embedding/embedding_weights/readVinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/Uniqueeinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
Tindices0	*
Taxis0*
Tparams0*

batch_dims *'
_output_shapes
:         
Й
iinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity`input_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
╜
Oinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparseSparseSegmentMeaniinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityXinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/Unique:1Tinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:         *

Tidx0
Ш
Ginput_layer/country_embedding/country_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
╕
Ainput_layer/country_embedding/country_embedding_weights/Reshape_1Reshapeainput_layer/country_embedding/country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2Ginput_layer/country_embedding/country_embedding_weights/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
╠
=input_layer/country_embedding/country_embedding_weights/ShapeShapeOinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
Х
Kinput_layer/country_embedding/country_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
Ч
Minput_layer/country_embedding/country_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Ч
Minput_layer/country_embedding/country_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
С
Einput_layer/country_embedding/country_embedding_weights/strided_sliceStridedSlice=input_layer/country_embedding/country_embedding_weights/ShapeKinput_layer/country_embedding/country_embedding_weights/strided_slice/stackMinput_layer/country_embedding/country_embedding_weights/strided_slice/stack_1Minput_layer/country_embedding/country_embedding_weights/strided_slice/stack_2*
Index0*
new_axis_mask *
ellipsis_mask *
shrink_axis_mask*

begin_mask *
end_mask *
T0*
_output_shapes
: 
Б
?input_layer/country_embedding/country_embedding_weights/stack/0Const*
_output_shapes
: *
value	B :*
dtype0
З
=input_layer/country_embedding/country_embedding_weights/stackPack?input_layer/country_embedding/country_embedding_weights/stack/0Einput_layer/country_embedding/country_embedding_weights/strided_slice*

axis *
N*
T0*
_output_shapes
:
У
<input_layer/country_embedding/country_embedding_weights/TileTileAinput_layer/country_embedding/country_embedding_weights/Reshape_1=input_layer/country_embedding/country_embedding_weights/stack*0
_output_shapes
:                  *
T0
*

Tmultiples0
╥
Binput_layer/country_embedding/country_embedding_weights/zeros_like	ZerosLikeOinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
╞
7input_layer/country_embedding/country_embedding_weightsSelect<input_layer/country_embedding/country_embedding_weights/TileBinput_layer/country_embedding/country_embedding_weights/zeros_likeOinput_layer/country_embedding/country_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┼
>input_layer/country_embedding/country_embedding_weights/Cast_1Cast9input_layer/country_embedding/to_sparse_input/dense_shape*
Truncate( *
_output_shapes
:*

DstT0*

SrcT0	
П
Einput_layer/country_embedding/country_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
О
Dinput_layer/country_embedding/country_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
╟
?input_layer/country_embedding/country_embedding_weights/Slice_1Slice>input_layer/country_embedding/country_embedding_weights/Cast_1Einput_layer/country_embedding/country_embedding_weights/Slice_1/beginDinput_layer/country_embedding/country_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
╢
?input_layer/country_embedding/country_embedding_weights/Shape_1Shape7input_layer/country_embedding/country_embedding_weights*
_output_shapes
:*
T0*
out_type0
П
Einput_layer/country_embedding/country_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
Ч
Dinput_layer/country_embedding/country_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
         *
dtype0
╚
?input_layer/country_embedding/country_embedding_weights/Slice_2Slice?input_layer/country_embedding/country_embedding_weights/Shape_1Einput_layer/country_embedding/country_embedding_weights/Slice_2/beginDinput_layer/country_embedding/country_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
Е
Cinput_layer/country_embedding/country_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╦
>input_layer/country_embedding/country_embedding_weights/concatConcatV2?input_layer/country_embedding/country_embedding_weights/Slice_1?input_layer/country_embedding/country_embedding_weights/Slice_2Cinput_layer/country_embedding/country_embedding_weights/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
Е
Ainput_layer/country_embedding/country_embedding_weights/Reshape_2Reshape7input_layer/country_embedding/country_embedding_weights>input_layer/country_embedding/country_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:         
д
#input_layer/country_embedding/ShapeShapeAinput_layer/country_embedding/country_embedding_weights/Reshape_2*
_output_shapes
:*
T0*
out_type0
{
1input_layer/country_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
}
3input_layer/country_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
}
3input_layer/country_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
П
+input_layer/country_embedding/strided_sliceStridedSlice#input_layer/country_embedding/Shape1input_layer/country_embedding/strided_slice/stack3input_layer/country_embedding/strided_slice/stack_13input_layer/country_embedding/strided_slice/stack_2*
Index0*
ellipsis_mask *
shrink_axis_mask*

begin_mask *
T0*
_output_shapes
: *
new_axis_mask *
end_mask 
o
-input_layer/country_embedding/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
╔
+input_layer/country_embedding/Reshape/shapePack+input_layer/country_embedding/strided_slice-input_layer/country_embedding/Reshape/shape/1*
N*

axis *
T0*
_output_shapes
:
р
%input_layer/country_embedding/ReshapeReshapeAinput_layer/country_embedding/country_embedding_weights/Reshape_2+input_layer/country_embedding/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:         
Y
input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
▌
input_layer/concatConcatV2,input_layer/cate_level1_id_embedding/Reshape,input_layer/cate_level2_id_embedding/Reshape,input_layer/cate_level3_id_embedding/Reshape,input_layer/cate_level4_id_embedding/Reshape%input_layer/country_embedding/Reshapeinput_layer/concat/axis*
T0*

Tidx0*
N*'
_output_shapes
:         $
Д
9input_layer_1/cart_7d_bucketized_embedding/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
5input_layer_1/cart_7d_bucketized_embedding/ExpandDims
ExpandDims%ParseSingleExample/ParseSingleExample9input_layer_1/cart_7d_bucketized_embedding/ExpandDims/dim*'
_output_shapes
:         *

Tdim0*
T0	
┐
/input_layer_1/cart_7d_bucketized_embedding/CastCast5input_layer_1/cart_7d_bucketized_embedding/ExpandDims*

SrcT0	*
Truncate( *'
_output_shapes
:         *

DstT0
М
4input_layer_1/cart_7d_bucketized_embedding/Bucketize	Bucketize/input_layer_1/cart_7d_bucketized_embedding/Cast*
T0*f

boundariesX
V"T      р@  РA  B  HB  ОB  ┬B  ■B  &C  SC АЕC АдC А╩C АЎC └D А3D └\D  ОD  ├D аE ╚E*'
_output_shapes
:         
д
0input_layer_1/cart_7d_bucketized_embedding/ShapeShape4input_layer_1/cart_7d_bucketized_embedding/Bucketize*
T0*
out_type0*
_output_shapes
:
И
>input_layer_1/cart_7d_bucketized_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
К
@input_layer_1/cart_7d_bucketized_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
К
@input_layer_1/cart_7d_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╨
8input_layer_1/cart_7d_bucketized_embedding/strided_sliceStridedSlice0input_layer_1/cart_7d_bucketized_embedding/Shape>input_layer_1/cart_7d_bucketized_embedding/strided_slice/stack@input_layer_1/cart_7d_bucketized_embedding/strided_slice/stack_1@input_layer_1/cart_7d_bucketized_embedding/strided_slice/stack_2*

begin_mask *
Index0*
T0*
end_mask *
_output_shapes
: *
new_axis_mask *
ellipsis_mask *
shrink_axis_mask
x
6input_layer_1/cart_7d_bucketized_embedding/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6input_layer_1/cart_7d_bucketized_embedding/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ф
0input_layer_1/cart_7d_bucketized_embedding/rangeRange6input_layer_1/cart_7d_bucketized_embedding/range/start8input_layer_1/cart_7d_bucketized_embedding/strided_slice6input_layer_1/cart_7d_bucketized_embedding/range/delta*#
_output_shapes
:         *

Tidx0
}
;input_layer_1/cart_7d_bucketized_embedding/ExpandDims_1/dimConst*
_output_shapes
: *
value	B :*
dtype0
Є
7input_layer_1/cart_7d_bucketized_embedding/ExpandDims_1
ExpandDims0input_layer_1/cart_7d_bucketized_embedding/range;input_layer_1/cart_7d_bucketized_embedding/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:         
К
9input_layer_1/cart_7d_bucketized_embedding/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
я
/input_layer_1/cart_7d_bucketized_embedding/TileTile7input_layer_1/cart_7d_bucketized_embedding/ExpandDims_19input_layer_1/cart_7d_bucketized_embedding/Tile/multiples*

Tmultiples0*'
_output_shapes
:         *
T0
Л
8input_layer_1/cart_7d_bucketized_embedding/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
ф
2input_layer_1/cart_7d_bucketized_embedding/ReshapeReshape/input_layer_1/cart_7d_bucketized_embedding/Tile8input_layer_1/cart_7d_bucketized_embedding/Reshape/shape*
T0*#
_output_shapes
:         *
Tshape0
z
8input_layer_1/cart_7d_bucketized_embedding/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
z
8input_layer_1/cart_7d_bucketized_embedding/range_1/limitConst*
value	B :*
_output_shapes
: *
dtype0
z
8input_layer_1/cart_7d_bucketized_embedding/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
С
2input_layer_1/cart_7d_bucketized_embedding/range_1Range8input_layer_1/cart_7d_bucketized_embedding/range_1/start8input_layer_1/cart_7d_bucketized_embedding/range_1/limit8input_layer_1/cart_7d_bucketized_embedding/range_1/delta*

Tidx0*
_output_shapes
:
╖
;input_layer_1/cart_7d_bucketized_embedding/Tile_1/multiplesPack8input_layer_1/cart_7d_bucketized_embedding/strided_slice*
N*
T0*

axis *
_output_shapes
:
ъ
1input_layer_1/cart_7d_bucketized_embedding/Tile_1Tile2input_layer_1/cart_7d_bucketized_embedding/range_1;input_layer_1/cart_7d_bucketized_embedding/Tile_1/multiples*#
_output_shapes
:         *

Tmultiples0*
T0
Н
:input_layer_1/cart_7d_bucketized_embedding/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
э
4input_layer_1/cart_7d_bucketized_embedding/Reshape_1Reshape4input_layer_1/cart_7d_bucketized_embedding/Bucketize:input_layer_1/cart_7d_bucketized_embedding/Reshape_1/shape*
T0*#
_output_shapes
:         *
Tshape0
r
0input_layer_1/cart_7d_bucketized_embedding/mul/xConst*
dtype0*
value	B :*
_output_shapes
: 
╚
.input_layer_1/cart_7d_bucketized_embedding/mulMul0input_layer_1/cart_7d_bucketized_embedding/mul/x1input_layer_1/cart_7d_bucketized_embedding/Tile_1*
T0*#
_output_shapes
:         
╦
.input_layer_1/cart_7d_bucketized_embedding/addAddV24input_layer_1/cart_7d_bucketized_embedding/Reshape_1.input_layer_1/cart_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
ц
0input_layer_1/cart_7d_bucketized_embedding/stackPack2input_layer_1/cart_7d_bucketized_embedding/Reshape1input_layer_1/cart_7d_bucketized_embedding/Tile_1*'
_output_shapes
:         *
N*

axis *
T0
К
9input_layer_1/cart_7d_bucketized_embedding/transpose/permConst*
valueB"       *
dtype0*
_output_shapes
:
э
4input_layer_1/cart_7d_bucketized_embedding/transpose	Transpose0input_layer_1/cart_7d_bucketized_embedding/stack9input_layer_1/cart_7d_bucketized_embedding/transpose/perm*
Tperm0*
T0*'
_output_shapes
:         
└
1input_layer_1/cart_7d_bucketized_embedding/Cast_1Cast4input_layer_1/cart_7d_bucketized_embedding/transpose*
Truncate( *'
_output_shapes
:         *

SrcT0*

DstT0	
v
4input_layer_1/cart_7d_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
ф
2input_layer_1/cart_7d_bucketized_embedding/stack_1Pack8input_layer_1/cart_7d_bucketized_embedding/strided_slice4input_layer_1/cart_7d_bucketized_embedding/stack_1/1*
_output_shapes
:*
T0*

axis *
N
▒
1input_layer_1/cart_7d_bucketized_embedding/Cast_2Cast2input_layer_1/cart_7d_bucketized_embedding/stack_1*
_output_shapes
:*

SrcT0*

DstT0	*
Truncate( 
Б
_input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *
dtype0*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights
Ї
^input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
valueB
 *    *
_output_shapes
: 
Ў
`input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights
¤
iinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal_input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
seed2 *O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
T0*
dtype0*
_output_shapes

:*

seed 
л
]input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMuliinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal`input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
T0*
_output_shapes

:
Щ
Yinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd]input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul^input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
Б
<input_layer_1/cart_7d_bucketized_embedding/embedding_weights
VariableV2*
	container *
shape
:*
shared_name *O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
dtype0
Й
Cinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/AssignAssign<input_layer_1/cart_7d_bucketized_embedding/embedding_weightsYinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
T0*
use_locking(*
validate_shape(*
_output_shapes

:*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights
Е
Ainput_layer_1/cart_7d_bucketized_embedding/embedding_weights/readIdentity<input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
T0*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
е
[input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
д
Zinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
№
Uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SliceSlice1input_layer_1/cart_7d_bucketized_embedding/Cast_2[input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice/beginZinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice/size*
Index0*
_output_shapes
:*
T0	
Я
Uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
╚
Tinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/ProdProdUinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SliceUinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
в
`input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Я
]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
о
Xinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2GatherV21input_layer_1/cart_7d_bucketized_embedding/Cast_2`input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2/indices]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2/axis*
Taxis0*
_output_shapes
: *
Tparams0	*
Tindices0*

batch_dims 
╚
Vinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Cast/xPackTinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/ProdXinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2*
T0	*

axis *
N*
_output_shapes
:
█
]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseReshapeSparseReshape1input_layer_1/cart_7d_bucketized_embedding/Cast_11input_layer_1/cart_7d_bucketized_embedding/Cast_2Vinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
╨
finput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity.input_layer_1/cart_7d_bucketized_embedding/add*
T0*#
_output_shapes
:         
а
^input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
т
\input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualfinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseReshape/Identity^input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GreaterEqual/y*#
_output_shapes
:         *
T0
ю
Uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/WhereWhere\input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

░
]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
╘
Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/ReshapeReshapeUinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Where]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape/shape*#
_output_shapes
:         *
Tshape0*
T0	
б
_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ц
Zinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_1GatherV2]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseReshapeWinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*
Taxis0*'
_output_shapes
:         *

batch_dims 
б
_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ы
Zinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_2GatherV2finput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseReshape/IdentityWinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_2/axis*

batch_dims *
Tparams0*
Tindices0	*#
_output_shapes
:         *
Taxis0
ъ
Xinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/IdentityIdentity_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
л
iinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B : *
dtype0
ъ
winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsZinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_1Zinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/GatherV2_2Xinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Identityiinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╠
{input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
╬
}input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╬
}input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ш
uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicewinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows{input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack}input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1}input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
ellipsis_mask *
T0	*
end_mask*
new_axis_mask *#
_output_shapes
:         *

begin_mask*
Index0*
shrink_axis_mask
╕
linput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastuinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*
Truncate( *

DstT0*

SrcT0	*#
_output_shapes
:         
┐
ninput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniqueyinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*
out_idx0*2
_output_shapes 
:         :         
Р
}input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *
value	B : *O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights
ю
xinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ainput_layer_1/cart_7d_bucketized_embedding/embedding_weights/readninput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique}input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0*'
_output_shapes
:         *
Taxis0*

batch_dims *
Tparams0*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights
║
Бinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityxinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:         
Ю
ginput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeanБinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitypinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1linput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:         
░
_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
А
Yinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_1Reshapeyinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2_input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
№
Uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/ShapeShapeginput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
н
cinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
п
einput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
п
einput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Й
]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_sliceStridedSliceUinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Shapecinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice/stackeinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice/stack_1einput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice/stack_2*
Index0*

begin_mask *
end_mask *
new_axis_mask *
ellipsis_mask *
shrink_axis_mask*
T0*
_output_shapes
: 
Щ
Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
╧
Uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/stackPackWinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/stack/0]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/strided_slice*
_output_shapes
:*
N*

axis *
T0
█
Tinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/TileTileYinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_1Uinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/stack*0
_output_shapes
:                  *
T0
*

Tmultiples0
В
Zinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/zeros_like	ZerosLikeginput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
ж
Oinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weightsSelectTinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/TileZinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/zeros_likeginput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╒
Vinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Cast_1Cast1input_layer_1/cart_7d_bucketized_embedding/Cast_2*

SrcT0	*
_output_shapes
:*

DstT0*
Truncate( 
з
]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
ж
\input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
з
Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_1SliceVinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Cast_1]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_1/begin\input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
ц
Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Shape_1ShapeOinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights*
out_type0*
_output_shapes
:*
T0
з
]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
п
\input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_2/sizeConst*
dtype0*
valueB:
         *
_output_shapes
:
и
Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_2SliceWinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Shape_1]input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_2/begin\input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Э
[input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
л
Vinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/concatConcatV2Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_1Winput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Slice_2[input_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
═
Yinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_2ReshapeOinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weightsVinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:         
╦
2input_layer_1/cart_7d_bucketized_embedding/Shape_1ShapeYinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_2*
out_type0*
_output_shapes
:*
T0
К
@input_layer_1/cart_7d_bucketized_embedding/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Binput_layer_1/cart_7d_bucketized_embedding/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
М
Binput_layer_1/cart_7d_bucketized_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
┌
:input_layer_1/cart_7d_bucketized_embedding/strided_slice_1StridedSlice2input_layer_1/cart_7d_bucketized_embedding/Shape_1@input_layer_1/cart_7d_bucketized_embedding/strided_slice_1/stackBinput_layer_1/cart_7d_bucketized_embedding/strided_slice_1/stack_1Binput_layer_1/cart_7d_bucketized_embedding/strided_slice_1/stack_2*

begin_mask *
new_axis_mask *
end_mask *
shrink_axis_mask*
Index0*
ellipsis_mask *
_output_shapes
: *
T0
~
<input_layer_1/cart_7d_bucketized_embedding/Reshape_2/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
Ў
:input_layer_1/cart_7d_bucketized_embedding/Reshape_2/shapePack:input_layer_1/cart_7d_bucketized_embedding/strided_slice_1<input_layer_1/cart_7d_bucketized_embedding/Reshape_2/shape/1*
T0*
_output_shapes
:*
N*

axis 
Ц
4input_layer_1/cart_7d_bucketized_embedding/Reshape_2ReshapeYinput_layer_1/cart_7d_bucketized_embedding/cart_7d_bucketized_embedding_weights/Reshape_2:input_layer_1/cart_7d_bucketized_embedding/Reshape_2/shape*'
_output_shapes
:         *
Tshape0*
T0
Е
:input_layer_1/click_7d_bucketized_embedding/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
ч
6input_layer_1/click_7d_bucketized_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_5:input_layer_1/click_7d_bucketized_embedding/ExpandDims/dim*
T0	*'
_output_shapes
:         *

Tdim0
┴
0input_layer_1/click_7d_bucketized_embedding/CastCast6input_layer_1/click_7d_bucketized_embedding/ExpandDims*
Truncate( *

DstT0*'
_output_shapes
:         *

SrcT0	
О
5input_layer_1/click_7d_bucketized_embedding/Bucketize	Bucketize0input_layer_1/click_7d_bucketized_embedding/Cast*
T0*f

boundariesX
V"T      .C А╜C @D @cD └ЫD  ╦D Р E  "E  GE ░oE ░ОE XжE и┬E xшE ╝F №$F рGF М|F ┤╣F pG*'
_output_shapes
:         
ж
1input_layer_1/click_7d_bucketized_embedding/ShapeShape5input_layer_1/click_7d_bucketized_embedding/Bucketize*
out_type0*
T0*
_output_shapes
:
Й
?input_layer_1/click_7d_bucketized_embedding/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
Л
Ainput_layer_1/click_7d_bucketized_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Л
Ainput_layer_1/click_7d_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╒
9input_layer_1/click_7d_bucketized_embedding/strided_sliceStridedSlice1input_layer_1/click_7d_bucketized_embedding/Shape?input_layer_1/click_7d_bucketized_embedding/strided_slice/stackAinput_layer_1/click_7d_bucketized_embedding/strided_slice/stack_1Ainput_layer_1/click_7d_bucketized_embedding/strided_slice/stack_2*
T0*
shrink_axis_mask*
Index0*
_output_shapes
: *
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask 
y
7input_layer_1/click_7d_bucketized_embedding/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
y
7input_layer_1/click_7d_bucketized_embedding/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ш
1input_layer_1/click_7d_bucketized_embedding/rangeRange7input_layer_1/click_7d_bucketized_embedding/range/start9input_layer_1/click_7d_bucketized_embedding/strided_slice7input_layer_1/click_7d_bucketized_embedding/range/delta*#
_output_shapes
:         *

Tidx0
~
<input_layer_1/click_7d_bucketized_embedding/ExpandDims_1/dimConst*
value	B :*
_output_shapes
: *
dtype0
ї
8input_layer_1/click_7d_bucketized_embedding/ExpandDims_1
ExpandDims1input_layer_1/click_7d_bucketized_embedding/range<input_layer_1/click_7d_bucketized_embedding/ExpandDims_1/dim*

Tdim0*'
_output_shapes
:         *
T0
Л
:input_layer_1/click_7d_bucketized_embedding/Tile/multiplesConst*
_output_shapes
:*
valueB"      *
dtype0
Є
0input_layer_1/click_7d_bucketized_embedding/TileTile8input_layer_1/click_7d_bucketized_embedding/ExpandDims_1:input_layer_1/click_7d_bucketized_embedding/Tile/multiples*
T0*'
_output_shapes
:         *

Tmultiples0
М
9input_layer_1/click_7d_bucketized_embedding/Reshape/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
ч
3input_layer_1/click_7d_bucketized_embedding/ReshapeReshape0input_layer_1/click_7d_bucketized_embedding/Tile9input_layer_1/click_7d_bucketized_embedding/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
{
9input_layer_1/click_7d_bucketized_embedding/range_1/startConst*
_output_shapes
: *
value	B : *
dtype0
{
9input_layer_1/click_7d_bucketized_embedding/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 
{
9input_layer_1/click_7d_bucketized_embedding/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Х
3input_layer_1/click_7d_bucketized_embedding/range_1Range9input_layer_1/click_7d_bucketized_embedding/range_1/start9input_layer_1/click_7d_bucketized_embedding/range_1/limit9input_layer_1/click_7d_bucketized_embedding/range_1/delta*
_output_shapes
:*

Tidx0
╣
<input_layer_1/click_7d_bucketized_embedding/Tile_1/multiplesPack9input_layer_1/click_7d_bucketized_embedding/strided_slice*
N*
T0*
_output_shapes
:*

axis 
э
2input_layer_1/click_7d_bucketized_embedding/Tile_1Tile3input_layer_1/click_7d_bucketized_embedding/range_1<input_layer_1/click_7d_bucketized_embedding/Tile_1/multiples*#
_output_shapes
:         *

Tmultiples0*
T0
О
;input_layer_1/click_7d_bucketized_embedding/Reshape_1/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
Ё
5input_layer_1/click_7d_bucketized_embedding/Reshape_1Reshape5input_layer_1/click_7d_bucketized_embedding/Bucketize;input_layer_1/click_7d_bucketized_embedding/Reshape_1/shape*
T0*
Tshape0*#
_output_shapes
:         
s
1input_layer_1/click_7d_bucketized_embedding/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
╦
/input_layer_1/click_7d_bucketized_embedding/mulMul1input_layer_1/click_7d_bucketized_embedding/mul/x2input_layer_1/click_7d_bucketized_embedding/Tile_1*#
_output_shapes
:         *
T0
╬
/input_layer_1/click_7d_bucketized_embedding/addAddV25input_layer_1/click_7d_bucketized_embedding/Reshape_1/input_layer_1/click_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
щ
1input_layer_1/click_7d_bucketized_embedding/stackPack3input_layer_1/click_7d_bucketized_embedding/Reshape2input_layer_1/click_7d_bucketized_embedding/Tile_1*
N*
T0*'
_output_shapes
:         *

axis 
Л
:input_layer_1/click_7d_bucketized_embedding/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
Ё
5input_layer_1/click_7d_bucketized_embedding/transpose	Transpose1input_layer_1/click_7d_bucketized_embedding/stack:input_layer_1/click_7d_bucketized_embedding/transpose/perm*'
_output_shapes
:         *
T0*
Tperm0
┬
2input_layer_1/click_7d_bucketized_embedding/Cast_1Cast5input_layer_1/click_7d_bucketized_embedding/transpose*
Truncate( *

SrcT0*

DstT0	*'
_output_shapes
:         
w
5input_layer_1/click_7d_bucketized_embedding/stack_1/1Const*
value	B :*
_output_shapes
: *
dtype0
ч
3input_layer_1/click_7d_bucketized_embedding/stack_1Pack9input_layer_1/click_7d_bucketized_embedding/strided_slice5input_layer_1/click_7d_bucketized_embedding/stack_1/1*
N*
_output_shapes
:*
T0*

axis 
│
2input_layer_1/click_7d_bucketized_embedding/Cast_2Cast3input_layer_1/click_7d_bucketized_embedding/stack_1*

SrcT0*
Truncate( *

DstT0	*
_output_shapes
:
Г
`input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
valueB"      *
_output_shapes
:*
dtype0
Ў
_input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
dtype0*
_output_shapes
: 
°
ainput_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
_output_shapes
: *
valueB
 *   ?*
dtype0
А
jinput_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*
seed2 *
dtype0*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*

seed 
п
^input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMuljinput_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalainput_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights
Э
Zinput_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd^input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul_input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
T0*
_output_shapes

:
Г
=input_layer_1/click_7d_bucketized_embedding/embedding_weights
VariableV2*
shared_name *
shape
:*
dtype0*
_output_shapes

:*
	container *P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights
Н
Dinput_layer_1/click_7d_bucketized_embedding/embedding_weights/AssignAssign=input_layer_1/click_7d_bucketized_embedding/embedding_weightsZinput_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
validate_shape(*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
use_locking(*
T0*
_output_shapes

:
И
Binput_layer_1/click_7d_bucketized_embedding/embedding_weights/readIdentity=input_layer_1/click_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*
T0
з
]input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
ж
\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Г
Winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SliceSlice2input_layer_1/click_7d_bucketized_embedding/Cast_2]input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice/begin\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
б
Winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
╬
Vinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/ProdProdWinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SliceWinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0	
д
binput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
б
_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╡
Zinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2GatherV22input_layer_1/click_7d_bucketized_embedding/Cast_2binput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2/indices_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*

batch_dims *
_output_shapes
: *
Tparams0	
╬
Xinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Cast/xPackVinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/ProdZinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2*
_output_shapes
:*
N*

axis *
T0	
с
_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseReshapeSparseReshape2input_layer_1/click_7d_bucketized_embedding/Cast_12input_layer_1/click_7d_bucketized_embedding/Cast_2Xinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
╙
hinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity/input_layer_1/click_7d_bucketized_embedding/add*
T0*#
_output_shapes
:         
в
`input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
ш
^input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualhinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseReshape/Identity`input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GreaterEqual/y*
T0*#
_output_shapes
:         
Є
Winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/WhereWhere^input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

▓
_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
┌
Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/ReshapeReshapeWinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Where_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape/shape*#
_output_shapes
:         *
T0	*
Tshape0
г
ainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ю
\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_1GatherV2_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseReshapeYinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshapeainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*'
_output_shapes
:         *

batch_dims *
Tparams0	
г
ainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
є
\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_2GatherV2hinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseReshape/IdentityYinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshapeainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_2/axis*#
_output_shapes
:         *
Tparams0*

batch_dims *
Taxis0*
Tindices0	
ю
Zinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/IdentityIdentityainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
н
kinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
value	B : *
dtype0
Ї
yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_1\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/GatherV2_2Zinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Identitykinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╬
}input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
╨
input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
╨
input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
в
winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceyinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows}input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*

begin_mask*
ellipsis_mask *#
_output_shapes
:         *
Index0*
new_axis_mask *
end_mask*
shrink_axis_mask
╝
ninput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastwinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*
Truncate( *

SrcT0	*

DstT0*#
_output_shapes
:         
├
pinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUnique{input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*2
_output_shapes 
:         :         *
out_idx0
У
input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *
value	B : *
dtype0*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights
Ў
zinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Binput_layer_1/click_7d_bucketized_embedding/embedding_weights/readpinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/Uniqueinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:         *
Tindices0*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights*

batch_dims *
Tparams0*
Taxis0
╛
Гinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityzinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
ж
iinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeanГinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityrinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1ninput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:         *

Tidx0
▓
ainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       
Ж
[input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_1Reshape{input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ainput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_1/shape*'
_output_shapes
:         *
Tshape0*
T0

А
Winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/ShapeShapeiinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
п
einput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
▒
ginput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
▒
ginput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
У
_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_sliceStridedSliceWinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Shapeeinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice/stackginput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice/stack_1ginput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice/stack_2*
ellipsis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask 
Ы
Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/stack/0Const*
dtype0*
value	B :*
_output_shapes
: 
╒
Winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/stackPackYinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/stack/0_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/strided_slice*
T0*

axis *
_output_shapes
:*
N
с
Vinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/TileTile[input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_1Winput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
Ж
\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/zeros_like	ZerosLikeiinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
о
Qinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weightsSelectVinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Tile\input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/zeros_likeiinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╪
Xinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Cast_1Cast2input_layer_1/click_7d_bucketized_embedding/Cast_2*
_output_shapes
:*

DstT0*
Truncate( *

SrcT0	
й
_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
и
^input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
п
Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_1SliceXinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Cast_1_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_1/begin^input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
ъ
Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Shape_1ShapeQinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights*
out_type0*
T0*
_output_shapes
:
й
_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
valueB:*
dtype0
▒
^input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
         
░
Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_2SliceYinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Shape_1_input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_2/begin^input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Я
]input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
│
Xinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/concatConcatV2Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_1Yinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Slice_2]input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/concat/axis*
_output_shapes
:*
N*

Tidx0*
T0
╙
[input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_2ReshapeQinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weightsXinput_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/concat*
Tshape0*
T0*'
_output_shapes
:         
╬
3input_layer_1/click_7d_bucketized_embedding/Shape_1Shape[input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_2*
out_type0*
T0*
_output_shapes
:
Л
Ainput_layer_1/click_7d_bucketized_embedding/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
Н
Cinput_layer_1/click_7d_bucketized_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Н
Cinput_layer_1/click_7d_bucketized_embedding/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
▀
;input_layer_1/click_7d_bucketized_embedding/strided_slice_1StridedSlice3input_layer_1/click_7d_bucketized_embedding/Shape_1Ainput_layer_1/click_7d_bucketized_embedding/strided_slice_1/stackCinput_layer_1/click_7d_bucketized_embedding/strided_slice_1/stack_1Cinput_layer_1/click_7d_bucketized_embedding/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
_output_shapes
: *
end_mask *

begin_mask *
T0*
new_axis_mask *
ellipsis_mask 

=input_layer_1/click_7d_bucketized_embedding/Reshape_2/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
∙
;input_layer_1/click_7d_bucketized_embedding/Reshape_2/shapePack;input_layer_1/click_7d_bucketized_embedding/strided_slice_1=input_layer_1/click_7d_bucketized_embedding/Reshape_2/shape/1*

axis *
_output_shapes
:*
T0*
N
Ъ
5input_layer_1/click_7d_bucketized_embedding/Reshape_2Reshape[input_layer_1/click_7d_bucketized_embedding/click_7d_bucketized_embedding_weights/Reshape_2;input_layer_1/click_7d_bucketized_embedding/Reshape_2/shape*'
_output_shapes
:         *
Tshape0*
T0
Г
8input_layer_1/ctr_7d_bucketized_embedding/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
у
4input_layer_1/ctr_7d_bucketized_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_78input_layer_1/ctr_7d_bucketized_embedding/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
М
3input_layer_1/ctr_7d_bucketized_embedding/Bucketize	Bucketize4input_layer_1/ctr_7d_bucketized_embedding/ExpandDims*
T0*'
_output_shapes
:         *b

boundariesT
R"PhСm<■╖Т<DQа<яцй<+┘▒<╬е╕<╨a╛<╩2─<щЪ╔<B[╬<Rэ╙<jj┘<]P▀<,Ях<Ы■ь<ю▒Ї<п■<&S=┼=/:>
в
/input_layer_1/ctr_7d_bucketized_embedding/ShapeShape3input_layer_1/ctr_7d_bucketized_embedding/Bucketize*
_output_shapes
:*
T0*
out_type0
З
=input_layer_1/ctr_7d_bucketized_embedding/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Й
?input_layer_1/ctr_7d_bucketized_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Й
?input_layer_1/ctr_7d_bucketized_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╦
7input_layer_1/ctr_7d_bucketized_embedding/strided_sliceStridedSlice/input_layer_1/ctr_7d_bucketized_embedding/Shape=input_layer_1/ctr_7d_bucketized_embedding/strided_slice/stack?input_layer_1/ctr_7d_bucketized_embedding/strided_slice/stack_1?input_layer_1/ctr_7d_bucketized_embedding/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *
new_axis_mask *
T0*
_output_shapes
: *
Index0*
end_mask *

begin_mask 
w
5input_layer_1/ctr_7d_bucketized_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
w
5input_layer_1/ctr_7d_bucketized_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Р
/input_layer_1/ctr_7d_bucketized_embedding/rangeRange5input_layer_1/ctr_7d_bucketized_embedding/range/start7input_layer_1/ctr_7d_bucketized_embedding/strided_slice5input_layer_1/ctr_7d_bucketized_embedding/range/delta*

Tidx0*#
_output_shapes
:         
|
:input_layer_1/ctr_7d_bucketized_embedding/ExpandDims_1/dimConst*
dtype0*
value	B :*
_output_shapes
: 
я
6input_layer_1/ctr_7d_bucketized_embedding/ExpandDims_1
ExpandDims/input_layer_1/ctr_7d_bucketized_embedding/range:input_layer_1/ctr_7d_bucketized_embedding/ExpandDims_1/dim*

Tdim0*'
_output_shapes
:         *
T0
Й
8input_layer_1/ctr_7d_bucketized_embedding/Tile/multiplesConst*
valueB"      *
_output_shapes
:*
dtype0
ь
.input_layer_1/ctr_7d_bucketized_embedding/TileTile6input_layer_1/ctr_7d_bucketized_embedding/ExpandDims_18input_layer_1/ctr_7d_bucketized_embedding/Tile/multiples*'
_output_shapes
:         *
T0*

Tmultiples0
К
7input_layer_1/ctr_7d_bucketized_embedding/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
с
1input_layer_1/ctr_7d_bucketized_embedding/ReshapeReshape.input_layer_1/ctr_7d_bucketized_embedding/Tile7input_layer_1/ctr_7d_bucketized_embedding/Reshape/shape*
T0*#
_output_shapes
:         *
Tshape0
y
7input_layer_1/ctr_7d_bucketized_embedding/range_1/startConst*
value	B : *
_output_shapes
: *
dtype0
y
7input_layer_1/ctr_7d_bucketized_embedding/range_1/limitConst*
_output_shapes
: *
value	B :*
dtype0
y
7input_layer_1/ctr_7d_bucketized_embedding/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Н
1input_layer_1/ctr_7d_bucketized_embedding/range_1Range7input_layer_1/ctr_7d_bucketized_embedding/range_1/start7input_layer_1/ctr_7d_bucketized_embedding/range_1/limit7input_layer_1/ctr_7d_bucketized_embedding/range_1/delta*
_output_shapes
:*

Tidx0
╡
:input_layer_1/ctr_7d_bucketized_embedding/Tile_1/multiplesPack7input_layer_1/ctr_7d_bucketized_embedding/strided_slice*
_output_shapes
:*
T0*

axis *
N
ч
0input_layer_1/ctr_7d_bucketized_embedding/Tile_1Tile1input_layer_1/ctr_7d_bucketized_embedding/range_1:input_layer_1/ctr_7d_bucketized_embedding/Tile_1/multiples*

Tmultiples0*#
_output_shapes
:         *
T0
М
9input_layer_1/ctr_7d_bucketized_embedding/Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ъ
3input_layer_1/ctr_7d_bucketized_embedding/Reshape_1Reshape3input_layer_1/ctr_7d_bucketized_embedding/Bucketize9input_layer_1/ctr_7d_bucketized_embedding/Reshape_1/shape*#
_output_shapes
:         *
T0*
Tshape0
q
/input_layer_1/ctr_7d_bucketized_embedding/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
┼
-input_layer_1/ctr_7d_bucketized_embedding/mulMul/input_layer_1/ctr_7d_bucketized_embedding/mul/x0input_layer_1/ctr_7d_bucketized_embedding/Tile_1*#
_output_shapes
:         *
T0
╚
-input_layer_1/ctr_7d_bucketized_embedding/addAddV23input_layer_1/ctr_7d_bucketized_embedding/Reshape_1-input_layer_1/ctr_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
у
/input_layer_1/ctr_7d_bucketized_embedding/stackPack1input_layer_1/ctr_7d_bucketized_embedding/Reshape0input_layer_1/ctr_7d_bucketized_embedding/Tile_1*
N*'
_output_shapes
:         *
T0*

axis 
Й
8input_layer_1/ctr_7d_bucketized_embedding/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
ъ
3input_layer_1/ctr_7d_bucketized_embedding/transpose	Transpose/input_layer_1/ctr_7d_bucketized_embedding/stack8input_layer_1/ctr_7d_bucketized_embedding/transpose/perm*'
_output_shapes
:         *
Tperm0*
T0
╝
.input_layer_1/ctr_7d_bucketized_embedding/CastCast3input_layer_1/ctr_7d_bucketized_embedding/transpose*
Truncate( *

SrcT0*'
_output_shapes
:         *

DstT0	
u
3input_layer_1/ctr_7d_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
с
1input_layer_1/ctr_7d_bucketized_embedding/stack_1Pack7input_layer_1/ctr_7d_bucketized_embedding/strided_slice3input_layer_1/ctr_7d_bucketized_embedding/stack_1/1*
N*
_output_shapes
:*
T0*

axis 
п
0input_layer_1/ctr_7d_bucketized_embedding/Cast_1Cast1input_layer_1/ctr_7d_bucketized_embedding/stack_1*
Truncate( *

SrcT0*

DstT0	*
_output_shapes
:
 
^input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"      *
dtype0*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
_output_shapes
:
Є
]input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
valueB
 *    
Ї
_input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
valueB
 *   ?
·
hinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal^input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
seed2 *N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*

seed *
T0*
_output_shapes

:*
dtype0
з
\input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMulhinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal_input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes

:*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights
Х
Xinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd\input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul]input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
 
;input_layer_1/ctr_7d_bucketized_embedding/embedding_weights
VariableV2*
	container *
_output_shapes

:*
dtype0*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
shape
:*
shared_name 
Е
Binput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/AssignAssign;input_layer_1/ctr_7d_bucketized_embedding/embedding_weightsXinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
use_locking(*
T0*
_output_shapes

:*
validate_shape(*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights
В
@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/readIdentity;input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights
г
Yinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
valueB: *
dtype0
в
Xinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
ї
Sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SliceSlice0input_layer_1/ctr_7d_bucketized_embedding/Cast_1Yinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice/beginXinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
Э
Sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
┬
Rinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/ProdProdSinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SliceSinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Const*

Tidx0*
T0	*
_output_shapes
: *
	keep_dims( 
а
^input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Э
[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
з
Vinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2GatherV20input_layer_1/ctr_7d_bucketized_embedding/Cast_1^input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2/indices[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2/axis*
Tindices0*
_output_shapes
: *
Taxis0*
Tparams0	*

batch_dims 
┬
Tinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Cast/xPackRinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/ProdVinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	*

axis 
╙
[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseReshapeSparseReshape.input_layer_1/ctr_7d_bucketized_embedding/Cast0input_layer_1/ctr_7d_bucketized_embedding/Cast_1Tinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
═
dinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity-input_layer_1/ctr_7d_bucketized_embedding/add*#
_output_shapes
:         *
T0
Ю
\input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B : *
dtype0
▄
Zinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualdinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseReshape/Identity\input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GreaterEqual/y*#
_output_shapes
:         *
T0
ъ
Sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/WhereWhereZinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

о
[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╬
Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/ReshapeReshapeSinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Where[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:         *
Tshape0
Я
]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
▐
Xinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_1GatherV2[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseReshapeUinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:         *
Taxis0*
Tindices0	*

batch_dims 
Я
]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
у
Xinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_2GatherV2dinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseReshape/IdentityUinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_2/axis*

batch_dims *
Tparams0*#
_output_shapes
:         *
Tindices0	*
Taxis0
ц
Vinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/IdentityIdentity]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
й
ginput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0*
value	B : *
_output_shapes
: 
р
uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_1Xinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/GatherV2_2Vinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Identityginput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*
T0*T
_output_shapesB
@:         :         :         :         
╩
yinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
_output_shapes
:*
dtype0
╠
{input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
╠
{input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
О
sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*#
_output_shapes
:         *
Index0*
ellipsis_mask *
new_axis_mask *

begin_mask*
end_mask*
shrink_axis_mask*
T0	
┤
jinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*
Truncate( *

DstT0*

SrcT0	*#
_output_shapes
:         
╗
linput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*
out_idx0*2
_output_shapes 
:         :         
Н
{input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
value	B : *
dtype0
ц
vinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights/readlinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique{input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights*
Tparams0*
Tindices0*

batch_dims *'
_output_shapes
:         *
Taxis0
╡
input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:         
Х
einput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*
T0*

Tidx0*'
_output_shapes
:         
о
]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
·
Winput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_1Reshapewinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_1/shape*
Tshape0*'
_output_shapes
:         *
T0

°
Sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/ShapeShapeeinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
л
ainput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
н
cinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
н
cinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_sliceStridedSliceSinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Shapeainput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice/stackcinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice/stack_1cinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice/stack_2*
T0*
new_axis_mask *
end_mask *
_output_shapes
: *
ellipsis_mask *
Index0*
shrink_axis_mask*

begin_mask 
Ч
Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/stack/0Const*
value	B :*
_output_shapes
: *
dtype0
╔
Sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/stackPackUinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/stack/0[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/strided_slice*

axis *
_output_shapes
:*
N*
T0
╒
Rinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/TileTileWinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_1Sinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
■
Xinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/zeros_like	ZerosLikeeinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
Ю
Minput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weightsSelectRinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/TileXinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/zeros_likeeinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╥
Tinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Cast_1Cast0input_layer_1/ctr_7d_bucketized_embedding/Cast_1*
_output_shapes
:*

DstT0*

SrcT0	*
Truncate( 
е
[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_1/beginConst*
dtype0*
valueB: *
_output_shapes
:
д
Zinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
Я
Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_1SliceTinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Cast_1[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_1/beginZinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_1/size*
_output_shapes
:*
T0*
Index0
т
Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Shape_1ShapeMinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights*
_output_shapes
:*
T0*
out_type0
е
[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
н
Zinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
         *
dtype0
а
Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_2SliceUinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Shape_1[input_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_2/beginZinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ы
Yinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
г
Tinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/concatConcatV2Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_1Uinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Slice_2Yinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
╟
Winput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_2ReshapeMinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weightsTinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/concat*
Tshape0*
T0*'
_output_shapes
:         
╚
1input_layer_1/ctr_7d_bucketized_embedding/Shape_1ShapeWinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_2*
_output_shapes
:*
T0*
out_type0
Й
?input_layer_1/ctr_7d_bucketized_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Л
Ainput_layer_1/ctr_7d_bucketized_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
Л
Ainput_layer_1/ctr_7d_bucketized_embedding/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╒
9input_layer_1/ctr_7d_bucketized_embedding/strided_slice_1StridedSlice1input_layer_1/ctr_7d_bucketized_embedding/Shape_1?input_layer_1/ctr_7d_bucketized_embedding/strided_slice_1/stackAinput_layer_1/ctr_7d_bucketized_embedding/strided_slice_1/stack_1Ainput_layer_1/ctr_7d_bucketized_embedding/strided_slice_1/stack_2*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
_output_shapes
: *
ellipsis_mask *
Index0*
new_axis_mask 
}
;input_layer_1/ctr_7d_bucketized_embedding/Reshape_2/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
є
9input_layer_1/ctr_7d_bucketized_embedding/Reshape_2/shapePack9input_layer_1/ctr_7d_bucketized_embedding/strided_slice_1;input_layer_1/ctr_7d_bucketized_embedding/Reshape_2/shape/1*
T0*
_output_shapes
:*
N*

axis 
Т
3input_layer_1/ctr_7d_bucketized_embedding/Reshape_2ReshapeWinput_layer_1/ctr_7d_bucketized_embedding/ctr_7d_bucketized_embedding_weights/Reshape_29input_layer_1/ctr_7d_bucketized_embedding/Reshape_2/shape*
Tshape0*
T0*'
_output_shapes
:         
Г
8input_layer_1/cvr_7d_bucketized_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
         
у
4input_layer_1/cvr_7d_bucketized_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_88input_layer_1/cvr_7d_bucketized_embedding/ExpandDims/dim*'
_output_shapes
:         *
T0*

Tdim0
М
3input_layer_1/cvr_7d_bucketized_embedding/Bucketize	Bucketize4input_layer_1/cvr_7d_bucketized_embedding/ExpandDims*b

boundariesT
R"P    ╜:╟:E/#;EїV;Б;╢gЦ;ж╕к;э╛;.V╘;5Fы;ЪФ<├Б<<f <m1<?F<L`<;╟А<%щЪ<√╦<  (B*
T0*'
_output_shapes
:         
в
/input_layer_1/cvr_7d_bucketized_embedding/ShapeShape3input_layer_1/cvr_7d_bucketized_embedding/Bucketize*
out_type0*
_output_shapes
:*
T0
З
=input_layer_1/cvr_7d_bucketized_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Й
?input_layer_1/cvr_7d_bucketized_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Й
?input_layer_1/cvr_7d_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╦
7input_layer_1/cvr_7d_bucketized_embedding/strided_sliceStridedSlice/input_layer_1/cvr_7d_bucketized_embedding/Shape=input_layer_1/cvr_7d_bucketized_embedding/strided_slice/stack?input_layer_1/cvr_7d_bucketized_embedding/strided_slice/stack_1?input_layer_1/cvr_7d_bucketized_embedding/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *

begin_mask 
w
5input_layer_1/cvr_7d_bucketized_embedding/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
w
5input_layer_1/cvr_7d_bucketized_embedding/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Р
/input_layer_1/cvr_7d_bucketized_embedding/rangeRange5input_layer_1/cvr_7d_bucketized_embedding/range/start7input_layer_1/cvr_7d_bucketized_embedding/strided_slice5input_layer_1/cvr_7d_bucketized_embedding/range/delta*

Tidx0*#
_output_shapes
:         
|
:input_layer_1/cvr_7d_bucketized_embedding/ExpandDims_1/dimConst*
_output_shapes
: *
value	B :*
dtype0
я
6input_layer_1/cvr_7d_bucketized_embedding/ExpandDims_1
ExpandDims/input_layer_1/cvr_7d_bucketized_embedding/range:input_layer_1/cvr_7d_bucketized_embedding/ExpandDims_1/dim*
T0*'
_output_shapes
:         *

Tdim0
Й
8input_layer_1/cvr_7d_bucketized_embedding/Tile/multiplesConst*
valueB"      *
_output_shapes
:*
dtype0
ь
.input_layer_1/cvr_7d_bucketized_embedding/TileTile6input_layer_1/cvr_7d_bucketized_embedding/ExpandDims_18input_layer_1/cvr_7d_bucketized_embedding/Tile/multiples*

Tmultiples0*'
_output_shapes
:         *
T0
К
7input_layer_1/cvr_7d_bucketized_embedding/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
с
1input_layer_1/cvr_7d_bucketized_embedding/ReshapeReshape.input_layer_1/cvr_7d_bucketized_embedding/Tile7input_layer_1/cvr_7d_bucketized_embedding/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
y
7input_layer_1/cvr_7d_bucketized_embedding/range_1/startConst*
_output_shapes
: *
value	B : *
dtype0
y
7input_layer_1/cvr_7d_bucketized_embedding/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 
y
7input_layer_1/cvr_7d_bucketized_embedding/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Н
1input_layer_1/cvr_7d_bucketized_embedding/range_1Range7input_layer_1/cvr_7d_bucketized_embedding/range_1/start7input_layer_1/cvr_7d_bucketized_embedding/range_1/limit7input_layer_1/cvr_7d_bucketized_embedding/range_1/delta*
_output_shapes
:*

Tidx0
╡
:input_layer_1/cvr_7d_bucketized_embedding/Tile_1/multiplesPack7input_layer_1/cvr_7d_bucketized_embedding/strided_slice*
T0*
N*
_output_shapes
:*

axis 
ч
0input_layer_1/cvr_7d_bucketized_embedding/Tile_1Tile1input_layer_1/cvr_7d_bucketized_embedding/range_1:input_layer_1/cvr_7d_bucketized_embedding/Tile_1/multiples*
T0*

Tmultiples0*#
_output_shapes
:         
М
9input_layer_1/cvr_7d_bucketized_embedding/Reshape_1/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
ъ
3input_layer_1/cvr_7d_bucketized_embedding/Reshape_1Reshape3input_layer_1/cvr_7d_bucketized_embedding/Bucketize9input_layer_1/cvr_7d_bucketized_embedding/Reshape_1/shape*
T0*#
_output_shapes
:         *
Tshape0
q
/input_layer_1/cvr_7d_bucketized_embedding/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
┼
-input_layer_1/cvr_7d_bucketized_embedding/mulMul/input_layer_1/cvr_7d_bucketized_embedding/mul/x0input_layer_1/cvr_7d_bucketized_embedding/Tile_1*
T0*#
_output_shapes
:         
╚
-input_layer_1/cvr_7d_bucketized_embedding/addAddV23input_layer_1/cvr_7d_bucketized_embedding/Reshape_1-input_layer_1/cvr_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
у
/input_layer_1/cvr_7d_bucketized_embedding/stackPack1input_layer_1/cvr_7d_bucketized_embedding/Reshape0input_layer_1/cvr_7d_bucketized_embedding/Tile_1*'
_output_shapes
:         *
N*
T0*

axis 
Й
8input_layer_1/cvr_7d_bucketized_embedding/transpose/permConst*
dtype0*
valueB"       *
_output_shapes
:
ъ
3input_layer_1/cvr_7d_bucketized_embedding/transpose	Transpose/input_layer_1/cvr_7d_bucketized_embedding/stack8input_layer_1/cvr_7d_bucketized_embedding/transpose/perm*
T0*
Tperm0*'
_output_shapes
:         
╝
.input_layer_1/cvr_7d_bucketized_embedding/CastCast3input_layer_1/cvr_7d_bucketized_embedding/transpose*
Truncate( *

DstT0	*

SrcT0*'
_output_shapes
:         
u
3input_layer_1/cvr_7d_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
с
1input_layer_1/cvr_7d_bucketized_embedding/stack_1Pack7input_layer_1/cvr_7d_bucketized_embedding/strided_slice3input_layer_1/cvr_7d_bucketized_embedding/stack_1/1*
N*

axis *
T0*
_output_shapes
:
п
0input_layer_1/cvr_7d_bucketized_embedding/Cast_1Cast1input_layer_1/cvr_7d_bucketized_embedding/stack_1*
_output_shapes
:*

SrcT0*
Truncate( *

DstT0	
 
^input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
valueB"      *
dtype0*
_output_shapes
:
Є
]input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
_output_shapes
: 
Ї
_input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
_output_shapes
: *
valueB
 *   ?
·
hinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal^input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*

seed *
T0*
dtype0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
seed2 *
_output_shapes

:
з
\input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMulhinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal_input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0
Х
Xinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd\input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul]input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
 
;input_layer_1/cvr_7d_bucketized_embedding/embedding_weights
VariableV2*
_output_shapes

:*
	container *
dtype0*
shared_name *
shape
:*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights
Е
Binput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/AssignAssign;input_layer_1/cvr_7d_bucketized_embedding/embedding_weightsXinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights
В
@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/readIdentity;input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights
г
Yinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
в
Xinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
ї
Sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SliceSlice0input_layer_1/cvr_7d_bucketized_embedding/Cast_1Yinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice/beginXinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Э
Sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
┬
Rinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/ProdProdSinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SliceSinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Const*
T0	*
	keep_dims( *

Tidx0*
_output_shapes
: 
а
^input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Э
[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
з
Vinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2GatherV20input_layer_1/cvr_7d_bucketized_embedding/Cast_1^input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2/indices[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2/axis*
_output_shapes
: *
Tindices0*
Taxis0*

batch_dims *
Tparams0	
┬
Tinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Cast/xPackRinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/ProdVinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2*

axis *
_output_shapes
:*
N*
T0	
╙
[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseReshapeSparseReshape.input_layer_1/cvr_7d_bucketized_embedding/Cast0input_layer_1/cvr_7d_bucketized_embedding/Cast_1Tinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
═
dinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity-input_layer_1/cvr_7d_bucketized_embedding/add*#
_output_shapes
:         *
T0
Ю
\input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GreaterEqual/yConst*
dtype0*
value	B : *
_output_shapes
: 
▄
Zinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualdinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseReshape/Identity\input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GreaterEqual/y*
T0*#
_output_shapes
:         
ъ
Sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/WhereWhereZinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

о
[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape/shapeConst*
dtype0*
valueB:
         *
_output_shapes
:
╬
Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/ReshapeReshapeSinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Where[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0	
Я
]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
▐
Xinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_1GatherV2[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseReshapeUinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_1/axis*
Tindices0	*
Taxis0*'
_output_shapes
:         *

batch_dims *
Tparams0	
Я
]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
у
Xinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_2GatherV2dinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseReshape/IdentityUinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_2/axis*
Tindices0	*

batch_dims *
Tparams0*
Taxis0*#
_output_shapes
:         
ц
Vinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/IdentityIdentity]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
й
ginput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B : *
_output_shapes
: *
dtype0
р
uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_1Xinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/GatherV2_2Vinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Identityginput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╩
yinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
╠
{input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
╠
{input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
О
sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*#
_output_shapes
:         *
T0	*

begin_mask*
Index0*
ellipsis_mask *
shrink_axis_mask*
new_axis_mask 
┤
jinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*#
_output_shapes
:         *

SrcT0	*

DstT0*
Truncate( 
╗
linput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*2
_output_shapes 
:         :         *
out_idx0
Н
{input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights*
value	B : *
dtype0
ц
vinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights/readlinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique{input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0*
Tparams0*'
_output_shapes
:         *

batch_dims *
Taxis0*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights
╡
input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
Х
einput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:         *

Tidx0
о
]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
·
Winput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_1Reshapewinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:         *
Tshape0
°
Sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/ShapeShapeeinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:*
out_type0
л
ainput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
н
cinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
н
cinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
 
[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_sliceStridedSliceSinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Shapeainput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice/stackcinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice/stack_1cinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
T0*
Index0*
_output_shapes
: *
new_axis_mask *
shrink_axis_mask*
end_mask 
Ч
Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
╔
Sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/stackPackUinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/stack/0[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/strided_slice*
_output_shapes
:*
T0*

axis *
N
╒
Rinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/TileTileWinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_1Sinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
■
Xinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/zeros_like	ZerosLikeeinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
Ю
Minput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weightsSelectRinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/TileXinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/zeros_likeeinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╥
Tinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Cast_1Cast0input_layer_1/cvr_7d_bucketized_embedding/Cast_1*

SrcT0	*

DstT0*
Truncate( *
_output_shapes
:
е
[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
д
Zinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Я
Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_1SliceTinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Cast_1[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_1/beginZinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_1/size*
_output_shapes
:*
Index0*
T0
т
Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Shape_1ShapeMinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights*
out_type0*
T0*
_output_shapes
:
е
[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
н
Zinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_2/sizeConst*
valueB:
         *
_output_shapes
:*
dtype0
а
Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_2SliceUinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Shape_1[input_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_2/beginZinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ы
Yinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
г
Tinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/concatConcatV2Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_1Uinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Slice_2Yinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
╟
Winput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_2ReshapeMinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weightsTinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/concat*'
_output_shapes
:         *
Tshape0*
T0
╚
1input_layer_1/cvr_7d_bucketized_embedding/Shape_1ShapeWinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
Й
?input_layer_1/cvr_7d_bucketized_embedding/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
Л
Ainput_layer_1/cvr_7d_bucketized_embedding/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Ainput_layer_1/cvr_7d_bucketized_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
╒
9input_layer_1/cvr_7d_bucketized_embedding/strided_slice_1StridedSlice1input_layer_1/cvr_7d_bucketized_embedding/Shape_1?input_layer_1/cvr_7d_bucketized_embedding/strided_slice_1/stackAinput_layer_1/cvr_7d_bucketized_embedding/strided_slice_1/stack_1Ainput_layer_1/cvr_7d_bucketized_embedding/strided_slice_1/stack_2*
Index0*
_output_shapes
: *

begin_mask *
end_mask *
T0*
ellipsis_mask *
shrink_axis_mask*
new_axis_mask 
}
;input_layer_1/cvr_7d_bucketized_embedding/Reshape_2/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
є
9input_layer_1/cvr_7d_bucketized_embedding/Reshape_2/shapePack9input_layer_1/cvr_7d_bucketized_embedding/strided_slice_1;input_layer_1/cvr_7d_bucketized_embedding/Reshape_2/shape/1*

axis *
_output_shapes
:*
N*
T0
Т
3input_layer_1/cvr_7d_bucketized_embedding/Reshape_2ReshapeWinput_layer_1/cvr_7d_bucketized_embedding/cvr_7d_bucketized_embedding_weights/Reshape_29input_layer_1/cvr_7d_bucketized_embedding/Reshape_2/shape*
T0*'
_output_shapes
:         *
Tshape0
Г
8input_layer_1/ord_7d_bucketized_embedding/ExpandDims/dimConst*
valueB :
         *
_output_shapes
: *
dtype0
у
4input_layer_1/ord_7d_bucketized_embedding/ExpandDims
ExpandDims'ParseSingleExample/ParseSingleExample_98input_layer_1/ord_7d_bucketized_embedding/ExpandDims/dim*'
_output_shapes
:         *
T0	*

Tdim0
╜
.input_layer_1/ord_7d_bucketized_embedding/CastCast4input_layer_1/ord_7d_bucketized_embedding/ExpandDims*'
_output_shapes
:         *

DstT0*
Truncate( *

SrcT0	
К
3input_layer_1/ord_7d_bucketized_embedding/Bucketize	Bucketize.input_layer_1/ord_7d_bucketized_embedding/Cast*f

boundariesX
V"T  А┐      А?  @@  А@  р@  A  PA  ИA  ░A  ╪A  B  0B  \B  ЖB  дB  ╬B  C  9C АРC `ўD*'
_output_shapes
:         *
T0
в
/input_layer_1/ord_7d_bucketized_embedding/ShapeShape3input_layer_1/ord_7d_bucketized_embedding/Bucketize*
T0*
out_type0*
_output_shapes
:
З
=input_layer_1/ord_7d_bucketized_embedding/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Й
?input_layer_1/ord_7d_bucketized_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Й
?input_layer_1/ord_7d_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
╦
7input_layer_1/ord_7d_bucketized_embedding/strided_sliceStridedSlice/input_layer_1/ord_7d_bucketized_embedding/Shape=input_layer_1/ord_7d_bucketized_embedding/strided_slice/stack?input_layer_1/ord_7d_bucketized_embedding/strided_slice/stack_1?input_layer_1/ord_7d_bucketized_embedding/strided_slice/stack_2*
shrink_axis_mask*
end_mask *
T0*
Index0*
ellipsis_mask *
new_axis_mask *
_output_shapes
: *

begin_mask 
w
5input_layer_1/ord_7d_bucketized_embedding/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
w
5input_layer_1/ord_7d_bucketized_embedding/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Р
/input_layer_1/ord_7d_bucketized_embedding/rangeRange5input_layer_1/ord_7d_bucketized_embedding/range/start7input_layer_1/ord_7d_bucketized_embedding/strided_slice5input_layer_1/ord_7d_bucketized_embedding/range/delta*

Tidx0*#
_output_shapes
:         
|
:input_layer_1/ord_7d_bucketized_embedding/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
я
6input_layer_1/ord_7d_bucketized_embedding/ExpandDims_1
ExpandDims/input_layer_1/ord_7d_bucketized_embedding/range:input_layer_1/ord_7d_bucketized_embedding/ExpandDims_1/dim*'
_output_shapes
:         *
T0*

Tdim0
Й
8input_layer_1/ord_7d_bucketized_embedding/Tile/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
ь
.input_layer_1/ord_7d_bucketized_embedding/TileTile6input_layer_1/ord_7d_bucketized_embedding/ExpandDims_18input_layer_1/ord_7d_bucketized_embedding/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:         
К
7input_layer_1/ord_7d_bucketized_embedding/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
с
1input_layer_1/ord_7d_bucketized_embedding/ReshapeReshape.input_layer_1/ord_7d_bucketized_embedding/Tile7input_layer_1/ord_7d_bucketized_embedding/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
y
7input_layer_1/ord_7d_bucketized_embedding/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
y
7input_layer_1/ord_7d_bucketized_embedding/range_1/limitConst*
value	B :*
_output_shapes
: *
dtype0
y
7input_layer_1/ord_7d_bucketized_embedding/range_1/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Н
1input_layer_1/ord_7d_bucketized_embedding/range_1Range7input_layer_1/ord_7d_bucketized_embedding/range_1/start7input_layer_1/ord_7d_bucketized_embedding/range_1/limit7input_layer_1/ord_7d_bucketized_embedding/range_1/delta*
_output_shapes
:*

Tidx0
╡
:input_layer_1/ord_7d_bucketized_embedding/Tile_1/multiplesPack7input_layer_1/ord_7d_bucketized_embedding/strided_slice*

axis *
N*
_output_shapes
:*
T0
ч
0input_layer_1/ord_7d_bucketized_embedding/Tile_1Tile1input_layer_1/ord_7d_bucketized_embedding/range_1:input_layer_1/ord_7d_bucketized_embedding/Tile_1/multiples*#
_output_shapes
:         *

Tmultiples0*
T0
М
9input_layer_1/ord_7d_bucketized_embedding/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
ъ
3input_layer_1/ord_7d_bucketized_embedding/Reshape_1Reshape3input_layer_1/ord_7d_bucketized_embedding/Bucketize9input_layer_1/ord_7d_bucketized_embedding/Reshape_1/shape*
Tshape0*#
_output_shapes
:         *
T0
q
/input_layer_1/ord_7d_bucketized_embedding/mul/xConst*
_output_shapes
: *
value	B :*
dtype0
┼
-input_layer_1/ord_7d_bucketized_embedding/mulMul/input_layer_1/ord_7d_bucketized_embedding/mul/x0input_layer_1/ord_7d_bucketized_embedding/Tile_1*#
_output_shapes
:         *
T0
╚
-input_layer_1/ord_7d_bucketized_embedding/addAddV23input_layer_1/ord_7d_bucketized_embedding/Reshape_1-input_layer_1/ord_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
у
/input_layer_1/ord_7d_bucketized_embedding/stackPack1input_layer_1/ord_7d_bucketized_embedding/Reshape0input_layer_1/ord_7d_bucketized_embedding/Tile_1*
T0*
N*'
_output_shapes
:         *

axis 
Й
8input_layer_1/ord_7d_bucketized_embedding/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
ъ
3input_layer_1/ord_7d_bucketized_embedding/transpose	Transpose/input_layer_1/ord_7d_bucketized_embedding/stack8input_layer_1/ord_7d_bucketized_embedding/transpose/perm*
Tperm0*
T0*'
_output_shapes
:         
╛
0input_layer_1/ord_7d_bucketized_embedding/Cast_1Cast3input_layer_1/ord_7d_bucketized_embedding/transpose*

DstT0	*
Truncate( *'
_output_shapes
:         *

SrcT0
u
3input_layer_1/ord_7d_bucketized_embedding/stack_1/1Const*
dtype0*
_output_shapes
: *
value	B :
с
1input_layer_1/ord_7d_bucketized_embedding/stack_1Pack7input_layer_1/ord_7d_bucketized_embedding/strided_slice3input_layer_1/ord_7d_bucketized_embedding/stack_1/1*
N*
T0*

axis *
_output_shapes
:
п
0input_layer_1/ord_7d_bucketized_embedding/Cast_2Cast1input_layer_1/ord_7d_bucketized_embedding/stack_1*
_output_shapes
:*

SrcT0*
Truncate( *

DstT0	
 
^input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
valueB"      
Є
]input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
valueB
 *    
Ї
_input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
valueB
 *   ?
·
hinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal^input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*

seed *N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
seed2 *
T0*
_output_shapes

:*
dtype0
з
\input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMulhinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal_input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
Х
Xinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd\input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul]input_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*
_output_shapes

:*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights
 
;input_layer_1/ord_7d_bucketized_embedding/embedding_weights
VariableV2*
dtype0*
	container *N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
shape
:*
shared_name 
Е
Binput_layer_1/ord_7d_bucketized_embedding/embedding_weights/AssignAssign;input_layer_1/ord_7d_bucketized_embedding/embedding_weightsXinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights
В
@input_layer_1/ord_7d_bucketized_embedding/embedding_weights/readIdentity;input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
T0
г
Yinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
в
Xinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
ї
Sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SliceSlice0input_layer_1/ord_7d_bucketized_embedding/Cast_2Yinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice/beginXinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Э
Sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/ConstConst*
_output_shapes
:*
valueB: *
dtype0
┬
Rinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/ProdProdSinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SliceSinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Const*
	keep_dims( *
T0	*
_output_shapes
: *

Tidx0
а
^input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
Э
[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
з
Vinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2GatherV20input_layer_1/ord_7d_bucketized_embedding/Cast_2^input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2/indices[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2/axis*
Tparams0	*
Tindices0*
Taxis0*
_output_shapes
: *

batch_dims 
┬
Tinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Cast/xPackRinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/ProdVinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2*

axis *
_output_shapes
:*
T0	*
N
╒
[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseReshapeSparseReshape0input_layer_1/ord_7d_bucketized_embedding/Cast_10input_layer_1/ord_7d_bucketized_embedding/Cast_2Tinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
═
dinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity-input_layer_1/ord_7d_bucketized_embedding/add*
T0*#
_output_shapes
:         
Ю
\input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
▄
Zinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualdinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseReshape/Identity\input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GreaterEqual/y*
T0*#
_output_shapes
:         
ъ
Sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/WhereWhereZinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

о
[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╬
Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/ReshapeReshapeSinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Where[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
Я
]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
▐
Xinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_1GatherV2[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseReshapeUinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:         *

batch_dims *
Tindices0	*
Taxis0
Я
]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
у
Xinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_2GatherV2dinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseReshape/IdentityUinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_2/axis*
Tparams0*
Tindices0	*
Taxis0*

batch_dims *#
_output_shapes
:         
ц
Vinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/IdentityIdentity]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
й
ginput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_1Xinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/GatherV2_2Vinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Identityginput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*
T0*T
_output_shapesB
@:         :         :         :         
╩
yinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
╠
{input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╠
{input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
О
sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
T0	*
new_axis_mask *#
_output_shapes
:         *
end_mask*
Index0
┤
jinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*
Truncate( *

DstT0*#
_output_shapes
:         *

SrcT0	
╗
linput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
out_idx0*2
_output_shapes 
:         :         *
T0
Н
{input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
dtype0*
_output_shapes
: *
value	B : 
ц
vinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2@input_layer_1/ord_7d_bucketized_embedding/embedding_weights/readlinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique{input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*

batch_dims *
Tparams0*
Tindices0*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*'
_output_shapes
:         *
Taxis0
╡
input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:         
Х
einput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*'
_output_shapes
:         *
T0
о
]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
valueB"       *
_output_shapes
:*
dtype0
·
Winput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_1Reshapewinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
°
Sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/ShapeShapeeinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
out_type0*
T0
л
ainput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
н
cinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
н
cinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
 
[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_sliceStridedSliceSinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Shapeainput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice/stackcinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice/stack_1cinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice/stack_2*
Index0*
_output_shapes
: *

begin_mask *
new_axis_mask *
shrink_axis_mask*
T0*
ellipsis_mask *
end_mask 
Ч
Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
╔
Sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/stackPackUinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/stack/0[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0*

axis 
╒
Rinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/TileTileWinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_1Sinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/stack*0
_output_shapes
:                  *
T0
*

Tmultiples0
■
Xinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/zeros_like	ZerosLikeeinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
Ю
Minput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weightsSelectRinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/TileXinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/zeros_likeeinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
╥
Tinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Cast_1Cast0input_layer_1/ord_7d_bucketized_embedding/Cast_2*

SrcT0	*

DstT0*
Truncate( *
_output_shapes
:
е
[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
д
Zinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Я
Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_1SliceTinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Cast_1[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_1/beginZinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
т
Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Shape_1ShapeMinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights*
_output_shapes
:*
T0*
out_type0
е
[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
н
Zinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
valueB:
         *
dtype0
а
Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_2SliceUinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Shape_1[input_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_2/beginZinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ы
Yinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
г
Tinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/concatConcatV2Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_1Uinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Slice_2Yinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
╟
Winput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_2ReshapeMinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weightsTinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/concat*'
_output_shapes
:         *
T0*
Tshape0
╚
1input_layer_1/ord_7d_bucketized_embedding/Shape_1ShapeWinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_2*
out_type0*
T0*
_output_shapes
:
Й
?input_layer_1/ord_7d_bucketized_embedding/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Л
Ainput_layer_1/ord_7d_bucketized_embedding/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
Л
Ainput_layer_1/ord_7d_bucketized_embedding/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╒
9input_layer_1/ord_7d_bucketized_embedding/strided_slice_1StridedSlice1input_layer_1/ord_7d_bucketized_embedding/Shape_1?input_layer_1/ord_7d_bucketized_embedding/strided_slice_1/stackAinput_layer_1/ord_7d_bucketized_embedding/strided_slice_1/stack_1Ainput_layer_1/ord_7d_bucketized_embedding/strided_slice_1/stack_2*

begin_mask *
T0*
_output_shapes
: *
shrink_axis_mask*
Index0*
ellipsis_mask *
new_axis_mask *
end_mask 
}
;input_layer_1/ord_7d_bucketized_embedding/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
є
9input_layer_1/ord_7d_bucketized_embedding/Reshape_2/shapePack9input_layer_1/ord_7d_bucketized_embedding/strided_slice_1;input_layer_1/ord_7d_bucketized_embedding/Reshape_2/shape/1*
_output_shapes
:*
T0*

axis *
N
Т
3input_layer_1/ord_7d_bucketized_embedding/Reshape_2ReshapeWinput_layer_1/ord_7d_bucketized_embedding/ord_7d_bucketized_embedding_weights/Reshape_29input_layer_1/ord_7d_bucketized_embedding/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
Ж
;input_layer_1/ord_total_bucketized_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
         
ъ
7input_layer_1/ord_total_bucketized_embedding/ExpandDims
ExpandDims(ParseSingleExample/ParseSingleExample_10;input_layer_1/ord_total_bucketized_embedding/ExpandDims/dim*

Tdim0*'
_output_shapes
:         *
T0	
├
1input_layer_1/ord_total_bucketized_embedding/CastCast7input_layer_1/ord_total_bucketized_embedding/ExpandDims*
Truncate( *

DstT0*

SrcT0	*'
_output_shapes
:         
Р
6input_layer_1/ord_total_bucketized_embedding/Bucketize	Bucketize1input_layer_1/ord_total_bucketized_embedding/Cast*'
_output_shapes
:         *f

boundariesX
V"T  А┐  А?  р@  АA  ╨A  B  `B  ЪB  ╨B  C  1C  iC  ЫC А╪C @D  [D `ЩD `ЄD └(E hЭE │бG*
T0
и
2input_layer_1/ord_total_bucketized_embedding/ShapeShape6input_layer_1/ord_total_bucketized_embedding/Bucketize*
out_type0*
_output_shapes
:*
T0
К
@input_layer_1/ord_total_bucketized_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
М
Binput_layer_1/ord_total_bucketized_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
М
Binput_layer_1/ord_total_bucketized_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
┌
:input_layer_1/ord_total_bucketized_embedding/strided_sliceStridedSlice2input_layer_1/ord_total_bucketized_embedding/Shape@input_layer_1/ord_total_bucketized_embedding/strided_slice/stackBinput_layer_1/ord_total_bucketized_embedding/strided_slice/stack_1Binput_layer_1/ord_total_bucketized_embedding/strided_slice/stack_2*
end_mask *
new_axis_mask *
shrink_axis_mask*

begin_mask *
Index0*
T0*
ellipsis_mask *
_output_shapes
: 
z
8input_layer_1/ord_total_bucketized_embedding/range/startConst*
value	B : *
_output_shapes
: *
dtype0
z
8input_layer_1/ord_total_bucketized_embedding/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Ь
2input_layer_1/ord_total_bucketized_embedding/rangeRange8input_layer_1/ord_total_bucketized_embedding/range/start:input_layer_1/ord_total_bucketized_embedding/strided_slice8input_layer_1/ord_total_bucketized_embedding/range/delta*

Tidx0*#
_output_shapes
:         

=input_layer_1/ord_total_bucketized_embedding/ExpandDims_1/dimConst*
_output_shapes
: *
value	B :*
dtype0
°
9input_layer_1/ord_total_bucketized_embedding/ExpandDims_1
ExpandDims2input_layer_1/ord_total_bucketized_embedding/range=input_layer_1/ord_total_bucketized_embedding/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:         
М
;input_layer_1/ord_total_bucketized_embedding/Tile/multiplesConst*
dtype0*
_output_shapes
:*
valueB"      
ї
1input_layer_1/ord_total_bucketized_embedding/TileTile9input_layer_1/ord_total_bucketized_embedding/ExpandDims_1;input_layer_1/ord_total_bucketized_embedding/Tile/multiples*'
_output_shapes
:         *
T0*

Tmultiples0
Н
:input_layer_1/ord_total_bucketized_embedding/Reshape/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
ъ
4input_layer_1/ord_total_bucketized_embedding/ReshapeReshape1input_layer_1/ord_total_bucketized_embedding/Tile:input_layer_1/ord_total_bucketized_embedding/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0
|
:input_layer_1/ord_total_bucketized_embedding/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
|
:input_layer_1/ord_total_bucketized_embedding/range_1/limitConst*
value	B :*
_output_shapes
: *
dtype0
|
:input_layer_1/ord_total_bucketized_embedding/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Щ
4input_layer_1/ord_total_bucketized_embedding/range_1Range:input_layer_1/ord_total_bucketized_embedding/range_1/start:input_layer_1/ord_total_bucketized_embedding/range_1/limit:input_layer_1/ord_total_bucketized_embedding/range_1/delta*

Tidx0*
_output_shapes
:
╗
=input_layer_1/ord_total_bucketized_embedding/Tile_1/multiplesPack:input_layer_1/ord_total_bucketized_embedding/strided_slice*
N*
_output_shapes
:*

axis *
T0
Ё
3input_layer_1/ord_total_bucketized_embedding/Tile_1Tile4input_layer_1/ord_total_bucketized_embedding/range_1=input_layer_1/ord_total_bucketized_embedding/Tile_1/multiples*
T0*#
_output_shapes
:         *

Tmultiples0
П
<input_layer_1/ord_total_bucketized_embedding/Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
є
6input_layer_1/ord_total_bucketized_embedding/Reshape_1Reshape6input_layer_1/ord_total_bucketized_embedding/Bucketize<input_layer_1/ord_total_bucketized_embedding/Reshape_1/shape*#
_output_shapes
:         *
T0*
Tshape0
t
2input_layer_1/ord_total_bucketized_embedding/mul/xConst*
value	B :*
_output_shapes
: *
dtype0
╬
0input_layer_1/ord_total_bucketized_embedding/mulMul2input_layer_1/ord_total_bucketized_embedding/mul/x3input_layer_1/ord_total_bucketized_embedding/Tile_1*#
_output_shapes
:         *
T0
╤
0input_layer_1/ord_total_bucketized_embedding/addAddV26input_layer_1/ord_total_bucketized_embedding/Reshape_10input_layer_1/ord_total_bucketized_embedding/mul*
T0*#
_output_shapes
:         
ь
2input_layer_1/ord_total_bucketized_embedding/stackPack4input_layer_1/ord_total_bucketized_embedding/Reshape3input_layer_1/ord_total_bucketized_embedding/Tile_1*
N*
T0*'
_output_shapes
:         *

axis 
М
;input_layer_1/ord_total_bucketized_embedding/transpose/permConst*
_output_shapes
:*
valueB"       *
dtype0
є
6input_layer_1/ord_total_bucketized_embedding/transpose	Transpose2input_layer_1/ord_total_bucketized_embedding/stack;input_layer_1/ord_total_bucketized_embedding/transpose/perm*
T0*'
_output_shapes
:         *
Tperm0
─
3input_layer_1/ord_total_bucketized_embedding/Cast_1Cast6input_layer_1/ord_total_bucketized_embedding/transpose*
Truncate( *

SrcT0*

DstT0	*'
_output_shapes
:         
x
6input_layer_1/ord_total_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
ъ
4input_layer_1/ord_total_bucketized_embedding/stack_1Pack:input_layer_1/ord_total_bucketized_embedding/strided_slice6input_layer_1/ord_total_bucketized_embedding/stack_1/1*

axis *
T0*
_output_shapes
:*
N
╡
3input_layer_1/ord_total_bucketized_embedding/Cast_2Cast4input_layer_1/ord_total_bucketized_embedding/stack_1*

DstT0	*

SrcT0*
Truncate( *
_output_shapes
:
Е
ainput_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"      *
_output_shapes
:*
dtype0*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights
°
`input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
valueB
 *    
·
binput_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *   ?*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
_output_shapes
: 
Г
kinput_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalainput_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
_output_shapes

:*
T0*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
seed2 *
dtype0*

seed 
│
_input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMulkinput_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalbinput_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*
T0*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
_output_shapes

:
б
[input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd_input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul`input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*
T0*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
_output_shapes

:
Е
>input_layer_1/ord_total_bucketized_embedding/embedding_weights
VariableV2*
shared_name *
	container *
dtype0*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
shape
:*
_output_shapes

:
С
Einput_layer_1/ord_total_bucketized_embedding/embedding_weights/AssignAssign>input_layer_1/ord_total_bucketized_embedding/embedding_weights[input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
T0*
use_locking(*
validate_shape(*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
_output_shapes

:
Л
Cinput_layer_1/ord_total_bucketized_embedding/embedding_weights/readIdentity>input_layer_1/ord_total_bucketized_embedding/embedding_weights*
T0*
_output_shapes

:*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights
й
_input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
и
^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
К
Yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SliceSlice3input_layer_1/ord_total_bucketized_embedding/Cast_2_input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice/begin^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice/size*
Index0*
_output_shapes
:*
T0	
г
Yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
╘
Xinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/ProdProdYinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SliceYinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Const*
T0	*

Tidx0*
	keep_dims( *
_output_shapes
: 
ж
dinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
г
ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
╝
\input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2GatherV23input_layer_1/ord_total_bucketized_embedding/Cast_2dinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2/indicesainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*

batch_dims *
Tindices0*
_output_shapes
: 
╘
Zinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Cast/xPackXinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Prod\input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2*
_output_shapes
:*

axis *
N*
T0	
ч
ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseReshapeSparseReshape3input_layer_1/ord_total_bucketized_embedding/Cast_13input_layer_1/ord_total_bucketized_embedding/Cast_2Zinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
╓
jinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseReshape/IdentityIdentity0input_layer_1/ord_total_bucketized_embedding/add*
T0*#
_output_shapes
:         
д
binput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
value	B : *
dtype0
ю
`input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GreaterEqualGreaterEqualjinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseReshape/Identitybinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GreaterEqual/y*
T0*#
_output_shapes
:         
Ў
Yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/WhereWhere`input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GreaterEqual*
T0
*'
_output_shapes
:         
┤
ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
р
[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/ReshapeReshapeYinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Whereainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0	
е
cinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ў
^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_1GatherV2ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseReshape[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshapecinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_1/axis*'
_output_shapes
:         *

batch_dims *
Tindices0	*
Taxis0*
Tparams0	
е
cinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
√
^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_2GatherV2jinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseReshape/Identity[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshapecinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_2/axis*

batch_dims *
Taxis0*#
_output_shapes
:         *
Tparams0*
Tindices0	
Є
\input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/IdentityIdentitycinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
п
minput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
■
{input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_1^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/GatherV2_2\input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Identityminput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╨
input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
╙
Бinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╙
Бinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
о
yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice{input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackБinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Бinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*
shrink_axis_mask*
Index0*
ellipsis_mask *#
_output_shapes
:         *

begin_mask*
T0	*
new_axis_mask 
└
pinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/CastCastyinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*
Truncate( *

DstT0*#
_output_shapes
:         
╟
rinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUnique}input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
out_idx0*2
_output_shapes 
:         :         *
T0
Ч
Бinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
_output_shapes
: *
dtype0*
value	B : 
 
|input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Cinput_layer_1/ord_total_bucketized_embedding/embedding_weights/readrinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/UniqueБinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*

batch_dims *Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*'
_output_shapes
:         *
Tindices0*
Taxis0*
Tparams0
┬
Еinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity|input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:         
о
kinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeanЕinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitytinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1pinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:         
┤
cinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
М
]input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_1Reshape}input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2cinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:         *
Tshape0
Д
Yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/ShapeShapekinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse*
out_type0*
T0*
_output_shapes
:
▒
ginput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
│
iinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
│
iinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Э
ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_sliceStridedSliceYinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Shapeginput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice/stackiinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice/stack_1iinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice/stack_2*
_output_shapes
: *
ellipsis_mask *
Index0*
T0*
new_axis_mask *
shrink_axis_mask*
end_mask *

begin_mask 
Э
[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
█
Yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/stackPack[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/stack/0ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/strided_slice*
_output_shapes
:*
T0*

axis *
N
ч
Xinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/TileTile]input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_1Yinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
К
^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/zeros_like	ZerosLikekinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╢
Sinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weightsSelectXinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Tile^input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/zeros_likekinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
█
Zinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Cast_1Cast3input_layer_1/ord_total_bucketized_embedding/Cast_2*

DstT0*

SrcT0	*
_output_shapes
:*
Truncate( 
л
ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
valueB: *
dtype0
к
`input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
╖
[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_1SliceZinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Cast_1ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_1/begin`input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ю
[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Shape_1ShapeSinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights*
out_type0*
T0*
_output_shapes
:
л
ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
│
`input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
╕
[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_2Slice[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Shape_1ainput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_2/begin`input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
б
_input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
╗
Zinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/concatConcatV2[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_1[input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Slice_2_input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/concat/axis*
T0*

Tidx0*
N*
_output_shapes
:
┘
]input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_2ReshapeSinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weightsZinput_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/concat*
Tshape0*'
_output_shapes
:         *
T0
╤
4input_layer_1/ord_total_bucketized_embedding/Shape_1Shape]input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_2*
out_type0*
T0*
_output_shapes
:
М
Binput_layer_1/ord_total_bucketized_embedding/strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0
О
Dinput_layer_1/ord_total_bucketized_embedding/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
О
Dinput_layer_1/ord_total_bucketized_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ф
<input_layer_1/ord_total_bucketized_embedding/strided_slice_1StridedSlice4input_layer_1/ord_total_bucketized_embedding/Shape_1Binput_layer_1/ord_total_bucketized_embedding/strided_slice_1/stackDinput_layer_1/ord_total_bucketized_embedding/strided_slice_1/stack_1Dinput_layer_1/ord_total_bucketized_embedding/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
T0*
end_mask *
shrink_axis_mask*
Index0*
_output_shapes
: *
new_axis_mask 
А
>input_layer_1/ord_total_bucketized_embedding/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
№
<input_layer_1/ord_total_bucketized_embedding/Reshape_2/shapePack<input_layer_1/ord_total_bucketized_embedding/strided_slice_1>input_layer_1/ord_total_bucketized_embedding/Reshape_2/shape/1*
T0*
_output_shapes
:*

axis *
N
Ю
6input_layer_1/ord_total_bucketized_embedding/Reshape_2Reshape]input_layer_1/ord_total_bucketized_embedding/ord_total_bucketized_embedding_weights/Reshape_2<input_layer_1/ord_total_bucketized_embedding/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
Г
8input_layer_1/pay_7d_bucketized_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
         *
dtype0
ф
4input_layer_1/pay_7d_bucketized_embedding/ExpandDims
ExpandDims(ParseSingleExample/ParseSingleExample_118input_layer_1/pay_7d_bucketized_embedding/ExpandDims/dim*

Tdim0*'
_output_shapes
:         *
T0	
╜
.input_layer_1/pay_7d_bucketized_embedding/CastCast4input_layer_1/pay_7d_bucketized_embedding/ExpandDims*

DstT0*'
_output_shapes
:         *
Truncate( *

SrcT0	
К
3input_layer_1/pay_7d_bucketized_embedding/Bucketize	Bucketize.input_layer_1/pay_7d_bucketized_embedding/Cast*f

boundariesX
V"T  А┐      А?   @  А@  └@   A  0A  pA  ШA  ╚A   B   B  DB  pB  ТB  ╕B  юB  'C  ВC аяD*
T0*'
_output_shapes
:         
в
/input_layer_1/pay_7d_bucketized_embedding/ShapeShape3input_layer_1/pay_7d_bucketized_embedding/Bucketize*
T0*
out_type0*
_output_shapes
:
З
=input_layer_1/pay_7d_bucketized_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Й
?input_layer_1/pay_7d_bucketized_embedding/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
Й
?input_layer_1/pay_7d_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
╦
7input_layer_1/pay_7d_bucketized_embedding/strided_sliceStridedSlice/input_layer_1/pay_7d_bucketized_embedding/Shape=input_layer_1/pay_7d_bucketized_embedding/strided_slice/stack?input_layer_1/pay_7d_bucketized_embedding/strided_slice/stack_1?input_layer_1/pay_7d_bucketized_embedding/strided_slice/stack_2*
ellipsis_mask *
_output_shapes
: *

begin_mask *
new_axis_mask *
T0*
end_mask *
Index0*
shrink_axis_mask
w
5input_layer_1/pay_7d_bucketized_embedding/range/startConst*
_output_shapes
: *
value	B : *
dtype0
w
5input_layer_1/pay_7d_bucketized_embedding/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Р
/input_layer_1/pay_7d_bucketized_embedding/rangeRange5input_layer_1/pay_7d_bucketized_embedding/range/start7input_layer_1/pay_7d_bucketized_embedding/strided_slice5input_layer_1/pay_7d_bucketized_embedding/range/delta*#
_output_shapes
:         *

Tidx0
|
:input_layer_1/pay_7d_bucketized_embedding/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
я
6input_layer_1/pay_7d_bucketized_embedding/ExpandDims_1
ExpandDims/input_layer_1/pay_7d_bucketized_embedding/range:input_layer_1/pay_7d_bucketized_embedding/ExpandDims_1/dim*'
_output_shapes
:         *

Tdim0*
T0
Й
8input_layer_1/pay_7d_bucketized_embedding/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
ь
.input_layer_1/pay_7d_bucketized_embedding/TileTile6input_layer_1/pay_7d_bucketized_embedding/ExpandDims_18input_layer_1/pay_7d_bucketized_embedding/Tile/multiples*

Tmultiples0*'
_output_shapes
:         *
T0
К
7input_layer_1/pay_7d_bucketized_embedding/Reshape/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
с
1input_layer_1/pay_7d_bucketized_embedding/ReshapeReshape.input_layer_1/pay_7d_bucketized_embedding/Tile7input_layer_1/pay_7d_bucketized_embedding/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
y
7input_layer_1/pay_7d_bucketized_embedding/range_1/startConst*
dtype0*
_output_shapes
: *
value	B : 
y
7input_layer_1/pay_7d_bucketized_embedding/range_1/limitConst*
value	B :*
_output_shapes
: *
dtype0
y
7input_layer_1/pay_7d_bucketized_embedding/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Н
1input_layer_1/pay_7d_bucketized_embedding/range_1Range7input_layer_1/pay_7d_bucketized_embedding/range_1/start7input_layer_1/pay_7d_bucketized_embedding/range_1/limit7input_layer_1/pay_7d_bucketized_embedding/range_1/delta*

Tidx0*
_output_shapes
:
╡
:input_layer_1/pay_7d_bucketized_embedding/Tile_1/multiplesPack7input_layer_1/pay_7d_bucketized_embedding/strided_slice*

axis *
T0*
_output_shapes
:*
N
ч
0input_layer_1/pay_7d_bucketized_embedding/Tile_1Tile1input_layer_1/pay_7d_bucketized_embedding/range_1:input_layer_1/pay_7d_bucketized_embedding/Tile_1/multiples*#
_output_shapes
:         *
T0*

Tmultiples0
М
9input_layer_1/pay_7d_bucketized_embedding/Reshape_1/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
ъ
3input_layer_1/pay_7d_bucketized_embedding/Reshape_1Reshape3input_layer_1/pay_7d_bucketized_embedding/Bucketize9input_layer_1/pay_7d_bucketized_embedding/Reshape_1/shape*
T0*
Tshape0*#
_output_shapes
:         
q
/input_layer_1/pay_7d_bucketized_embedding/mul/xConst*
_output_shapes
: *
value	B :*
dtype0
┼
-input_layer_1/pay_7d_bucketized_embedding/mulMul/input_layer_1/pay_7d_bucketized_embedding/mul/x0input_layer_1/pay_7d_bucketized_embedding/Tile_1*#
_output_shapes
:         *
T0
╚
-input_layer_1/pay_7d_bucketized_embedding/addAddV23input_layer_1/pay_7d_bucketized_embedding/Reshape_1-input_layer_1/pay_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
у
/input_layer_1/pay_7d_bucketized_embedding/stackPack1input_layer_1/pay_7d_bucketized_embedding/Reshape0input_layer_1/pay_7d_bucketized_embedding/Tile_1*
N*
T0*

axis *'
_output_shapes
:         
Й
8input_layer_1/pay_7d_bucketized_embedding/transpose/permConst*
valueB"       *
_output_shapes
:*
dtype0
ъ
3input_layer_1/pay_7d_bucketized_embedding/transpose	Transpose/input_layer_1/pay_7d_bucketized_embedding/stack8input_layer_1/pay_7d_bucketized_embedding/transpose/perm*
Tperm0*'
_output_shapes
:         *
T0
╛
0input_layer_1/pay_7d_bucketized_embedding/Cast_1Cast3input_layer_1/pay_7d_bucketized_embedding/transpose*

SrcT0*

DstT0	*'
_output_shapes
:         *
Truncate( 
u
3input_layer_1/pay_7d_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
с
1input_layer_1/pay_7d_bucketized_embedding/stack_1Pack7input_layer_1/pay_7d_bucketized_embedding/strided_slice3input_layer_1/pay_7d_bucketized_embedding/stack_1/1*
N*
_output_shapes
:*

axis *
T0
п
0input_layer_1/pay_7d_bucketized_embedding/Cast_2Cast1input_layer_1/pay_7d_bucketized_embedding/stack_1*
Truncate( *

SrcT0*
_output_shapes
:*

DstT0	
 
^input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
_output_shapes
:*
valueB"      *
dtype0
Є
]input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
_output_shapes
: *
dtype0*
valueB
 *    
Ї
_input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
valueB
 *   ?*
dtype0
·
hinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal^input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
_output_shapes

:*
seed2 *

seed *
dtype0*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
T0
з
\input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMulhinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal_input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
T0*
_output_shapes

:
Х
Xinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd\input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul]input_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0
 
;input_layer_1/pay_7d_bucketized_embedding/embedding_weights
VariableV2*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
	container *
shape
:*
_output_shapes

:*
dtype0*
shared_name 
Е
Binput_layer_1/pay_7d_bucketized_embedding/embedding_weights/AssignAssign;input_layer_1/pay_7d_bucketized_embedding/embedding_weightsXinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
validate_shape(*
T0*
use_locking(*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
В
@input_layer_1/pay_7d_bucketized_embedding/embedding_weights/readIdentity;input_layer_1/pay_7d_bucketized_embedding/embedding_weights*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0
г
Yinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
в
Xinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
ї
Sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SliceSlice0input_layer_1/pay_7d_bucketized_embedding/Cast_2Yinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice/beginXinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
Э
Sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
┬
Rinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/ProdProdSinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SliceSinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0	
а
^input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Э
[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
з
Vinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2GatherV20input_layer_1/pay_7d_bucketized_embedding/Cast_2^input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2/indices[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2/axis*

batch_dims *
Taxis0*
_output_shapes
: *
Tparams0	*
Tindices0
┬
Tinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Cast/xPackRinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/ProdVinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2*

axis *
N*
T0	*
_output_shapes
:
╒
[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseReshapeSparseReshape0input_layer_1/pay_7d_bucketized_embedding/Cast_10input_layer_1/pay_7d_bucketized_embedding/Cast_2Tinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
═
dinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity-input_layer_1/pay_7d_bucketized_embedding/add*#
_output_shapes
:         *
T0
Ю
\input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
▄
Zinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualdinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseReshape/Identity\input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GreaterEqual/y*#
_output_shapes
:         *
T0
ъ
Sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/WhereWhereZinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

о
[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
╬
Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/ReshapeReshapeSinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Where[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:         
Я
]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
▐
Xinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_1GatherV2[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseReshapeUinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_1/axis*
Tindices0	*

batch_dims *
Tparams0	*
Taxis0*'
_output_shapes
:         
Я
]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
value	B : *
dtype0
у
Xinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_2GatherV2dinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseReshape/IdentityUinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_2/axis*
Tparams0*#
_output_shapes
:         *
Tindices0	*

batch_dims *
Taxis0
ц
Vinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/IdentityIdentity]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
й
ginput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B : *
_output_shapes
: *
dtype0
р
uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsXinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_1Xinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/GatherV2_2Vinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Identityginput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╩
yinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
╠
{input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
╠
{input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
О
sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceuinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsyinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack{input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1{input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
new_axis_mask *

begin_mask*
end_mask*
shrink_axis_mask*
ellipsis_mask *
T0	*
Index0*#
_output_shapes
:         
┤
jinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastsinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:         
╗
linput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniquewinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*
out_idx0*2
_output_shapes 
:         :         
Н
{input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
value	B : 
ц
vinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2@input_layer_1/pay_7d_bucketized_embedding/embedding_weights/readlinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique{input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
Tparams0*

batch_dims *'
_output_shapes
:         *
Taxis0
╡
input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityvinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
Х
einput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeaninput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identityninput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1jinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*
T0*

Tidx0*'
_output_shapes
:         
о
]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
·
Winput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_1Reshapewinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2]input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_1/shape*'
_output_shapes
:         *
T0
*
Tshape0
°
Sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/ShapeShapeeinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:*
out_type0
л
ainput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
н
cinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
н
cinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_sliceStridedSliceSinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Shapeainput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice/stackcinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice/stack_1cinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice/stack_2*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
Index0*
_output_shapes
: *
shrink_axis_mask
Ч
Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
╔
Sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/stackPackUinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/stack/0[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N*

axis 
╒
Rinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/TileTileWinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_1Sinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/stack*0
_output_shapes
:                  *

Tmultiples0*
T0

■
Xinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/zeros_like	ZerosLikeeinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
Ю
Minput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weightsSelectRinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/TileXinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/zeros_likeeinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╥
Tinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Cast_1Cast0input_layer_1/pay_7d_bucketized_embedding/Cast_2*
_output_shapes
:*

SrcT0	*
Truncate( *

DstT0
е
[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
д
Zinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
Я
Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_1SliceTinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Cast_1[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_1/beginZinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
т
Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Shape_1ShapeMinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights*
_output_shapes
:*
T0*
out_type0
е
[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_2/beginConst*
dtype0*
valueB:*
_output_shapes
:
н
Zinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_2/sizeConst*
valueB:
         *
_output_shapes
:*
dtype0
а
Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_2SliceUinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Shape_1[input_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_2/beginZinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
Ы
Yinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
г
Tinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/concatConcatV2Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_1Uinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Slice_2Yinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
╟
Winput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_2ReshapeMinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weightsTinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/concat*
T0*'
_output_shapes
:         *
Tshape0
╚
1input_layer_1/pay_7d_bucketized_embedding/Shape_1ShapeWinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_2*
_output_shapes
:*
out_type0*
T0
Й
?input_layer_1/pay_7d_bucketized_embedding/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Л
Ainput_layer_1/pay_7d_bucketized_embedding/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Л
Ainput_layer_1/pay_7d_bucketized_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
╒
9input_layer_1/pay_7d_bucketized_embedding/strided_slice_1StridedSlice1input_layer_1/pay_7d_bucketized_embedding/Shape_1?input_layer_1/pay_7d_bucketized_embedding/strided_slice_1/stackAinput_layer_1/pay_7d_bucketized_embedding/strided_slice_1/stack_1Ainput_layer_1/pay_7d_bucketized_embedding/strided_slice_1/stack_2*
new_axis_mask *
shrink_axis_mask*
end_mask *
_output_shapes
: *

begin_mask *
T0*
Index0*
ellipsis_mask 
}
;input_layer_1/pay_7d_bucketized_embedding/Reshape_2/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
є
9input_layer_1/pay_7d_bucketized_embedding/Reshape_2/shapePack9input_layer_1/pay_7d_bucketized_embedding/strided_slice_1;input_layer_1/pay_7d_bucketized_embedding/Reshape_2/shape/1*

axis *
T0*
_output_shapes
:*
N
Т
3input_layer_1/pay_7d_bucketized_embedding/Reshape_2ReshapeWinput_layer_1/pay_7d_bucketized_embedding/pay_7d_bucketized_embedding_weights/Reshape_29input_layer_1/pay_7d_bucketized_embedding/Reshape_2/shape*'
_output_shapes
:         *
Tshape0*
T0
Ж
;input_layer_1/pay_total_bucketized_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         
ъ
7input_layer_1/pay_total_bucketized_embedding/ExpandDims
ExpandDims(ParseSingleExample/ParseSingleExample_12;input_layer_1/pay_total_bucketized_embedding/ExpandDims/dim*
T0	*'
_output_shapes
:         *

Tdim0
├
1input_layer_1/pay_total_bucketized_embedding/CastCast7input_layer_1/pay_total_bucketized_embedding/ExpandDims*

DstT0*

SrcT0	*
Truncate( *'
_output_shapes
:         
Р
6input_layer_1/pay_total_bucketized_embedding/Bucketize	Bucketize1input_layer_1/pay_total_bucketized_embedding/Cast*f

boundariesX
V"T  А┐  А?  └@  PA  ░A   B  0B  lB  ЮB  ╥B  C  1C  jC АЮC  ▐C └D А^D  гD 0 E Ё}E 6zG*'
_output_shapes
:         *
T0
и
2input_layer_1/pay_total_bucketized_embedding/ShapeShape6input_layer_1/pay_total_bucketized_embedding/Bucketize*
out_type0*
T0*
_output_shapes
:
К
@input_layer_1/pay_total_bucketized_embedding/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
М
Binput_layer_1/pay_total_bucketized_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
М
Binput_layer_1/pay_total_bucketized_embedding/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
┌
:input_layer_1/pay_total_bucketized_embedding/strided_sliceStridedSlice2input_layer_1/pay_total_bucketized_embedding/Shape@input_layer_1/pay_total_bucketized_embedding/strided_slice/stackBinput_layer_1/pay_total_bucketized_embedding/strided_slice/stack_1Binput_layer_1/pay_total_bucketized_embedding/strided_slice/stack_2*

begin_mask *
Index0*
shrink_axis_mask*
ellipsis_mask *
new_axis_mask *
T0*
_output_shapes
: *
end_mask 
z
8input_layer_1/pay_total_bucketized_embedding/range/startConst*
_output_shapes
: *
value	B : *
dtype0
z
8input_layer_1/pay_total_bucketized_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Ь
2input_layer_1/pay_total_bucketized_embedding/rangeRange8input_layer_1/pay_total_bucketized_embedding/range/start:input_layer_1/pay_total_bucketized_embedding/strided_slice8input_layer_1/pay_total_bucketized_embedding/range/delta*#
_output_shapes
:         *

Tidx0

=input_layer_1/pay_total_bucketized_embedding/ExpandDims_1/dimConst*
dtype0*
value	B :*
_output_shapes
: 
°
9input_layer_1/pay_total_bucketized_embedding/ExpandDims_1
ExpandDims2input_layer_1/pay_total_bucketized_embedding/range=input_layer_1/pay_total_bucketized_embedding/ExpandDims_1/dim*
T0*'
_output_shapes
:         *

Tdim0
М
;input_layer_1/pay_total_bucketized_embedding/Tile/multiplesConst*
_output_shapes
:*
valueB"      *
dtype0
ї
1input_layer_1/pay_total_bucketized_embedding/TileTile9input_layer_1/pay_total_bucketized_embedding/ExpandDims_1;input_layer_1/pay_total_bucketized_embedding/Tile/multiples*'
_output_shapes
:         *

Tmultiples0*
T0
Н
:input_layer_1/pay_total_bucketized_embedding/Reshape/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
ъ
4input_layer_1/pay_total_bucketized_embedding/ReshapeReshape1input_layer_1/pay_total_bucketized_embedding/Tile:input_layer_1/pay_total_bucketized_embedding/Reshape/shape*
Tshape0*#
_output_shapes
:         *
T0
|
:input_layer_1/pay_total_bucketized_embedding/range_1/startConst*
value	B : *
_output_shapes
: *
dtype0
|
:input_layer_1/pay_total_bucketized_embedding/range_1/limitConst*
value	B :*
_output_shapes
: *
dtype0
|
:input_layer_1/pay_total_bucketized_embedding/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
Щ
4input_layer_1/pay_total_bucketized_embedding/range_1Range:input_layer_1/pay_total_bucketized_embedding/range_1/start:input_layer_1/pay_total_bucketized_embedding/range_1/limit:input_layer_1/pay_total_bucketized_embedding/range_1/delta*
_output_shapes
:*

Tidx0
╗
=input_layer_1/pay_total_bucketized_embedding/Tile_1/multiplesPack:input_layer_1/pay_total_bucketized_embedding/strided_slice*
_output_shapes
:*
N*
T0*

axis 
Ё
3input_layer_1/pay_total_bucketized_embedding/Tile_1Tile4input_layer_1/pay_total_bucketized_embedding/range_1=input_layer_1/pay_total_bucketized_embedding/Tile_1/multiples*#
_output_shapes
:         *
T0*

Tmultiples0
П
<input_layer_1/pay_total_bucketized_embedding/Reshape_1/shapeConst*
valueB:
         *
_output_shapes
:*
dtype0
є
6input_layer_1/pay_total_bucketized_embedding/Reshape_1Reshape6input_layer_1/pay_total_bucketized_embedding/Bucketize<input_layer_1/pay_total_bucketized_embedding/Reshape_1/shape*
T0*
Tshape0*#
_output_shapes
:         
t
2input_layer_1/pay_total_bucketized_embedding/mul/xConst*
dtype0*
value	B :*
_output_shapes
: 
╬
0input_layer_1/pay_total_bucketized_embedding/mulMul2input_layer_1/pay_total_bucketized_embedding/mul/x3input_layer_1/pay_total_bucketized_embedding/Tile_1*
T0*#
_output_shapes
:         
╤
0input_layer_1/pay_total_bucketized_embedding/addAddV26input_layer_1/pay_total_bucketized_embedding/Reshape_10input_layer_1/pay_total_bucketized_embedding/mul*#
_output_shapes
:         *
T0
ь
2input_layer_1/pay_total_bucketized_embedding/stackPack4input_layer_1/pay_total_bucketized_embedding/Reshape3input_layer_1/pay_total_bucketized_embedding/Tile_1*
T0*

axis *
N*'
_output_shapes
:         
М
;input_layer_1/pay_total_bucketized_embedding/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
є
6input_layer_1/pay_total_bucketized_embedding/transpose	Transpose2input_layer_1/pay_total_bucketized_embedding/stack;input_layer_1/pay_total_bucketized_embedding/transpose/perm*
T0*
Tperm0*'
_output_shapes
:         
─
3input_layer_1/pay_total_bucketized_embedding/Cast_1Cast6input_layer_1/pay_total_bucketized_embedding/transpose*

DstT0	*'
_output_shapes
:         *

SrcT0*
Truncate( 
x
6input_layer_1/pay_total_bucketized_embedding/stack_1/1Const*
dtype0*
value	B :*
_output_shapes
: 
ъ
4input_layer_1/pay_total_bucketized_embedding/stack_1Pack:input_layer_1/pay_total_bucketized_embedding/strided_slice6input_layer_1/pay_total_bucketized_embedding/stack_1/1*
_output_shapes
:*
T0*

axis *
N
╡
3input_layer_1/pay_total_bucketized_embedding/Cast_2Cast4input_layer_1/pay_total_bucketized_embedding/stack_1*

DstT0	*
Truncate( *
_output_shapes
:*

SrcT0
Е
ainput_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
valueB"      *Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes
:*
dtype0
°
`input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    *Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights
·
binput_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
valueB
 *   ?*
dtype0*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes
: 
Г
kinput_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalainput_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
seed2 *Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes

:*

seed *
T0*
dtype0
│
_input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMulkinput_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalbinput_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes

:*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
T0
б
[input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd_input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul`input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0
Е
>input_layer_1/pay_total_bucketized_embedding/embedding_weights
VariableV2*
dtype0*
shape
:*
shared_name *
_output_shapes

:*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
	container 
С
Einput_layer_1/pay_total_bucketized_embedding/embedding_weights/AssignAssign>input_layer_1/pay_total_bucketized_embedding/embedding_weights[input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal*
validate_shape(*
T0*
_output_shapes

:*
use_locking(*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights
Л
Cinput_layer_1/pay_total_bucketized_embedding/embedding_weights/readIdentity>input_layer_1/pay_total_bucketized_embedding/embedding_weights*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0
й
_input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice/beginConst*
valueB: *
_output_shapes
:*
dtype0
и
^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
К
Yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SliceSlice3input_layer_1/pay_total_bucketized_embedding/Cast_2_input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice/begin^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice/size*
_output_shapes
:*
Index0*
T0	
г
Yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
╘
Xinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/ProdProdYinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SliceYinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Const*
T0	*

Tidx0*
_output_shapes
: *
	keep_dims( 
ж
dinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
value	B :*
dtype0
г
ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
╝
\input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2GatherV23input_layer_1/pay_total_bucketized_embedding/Cast_2dinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2/indicesainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2/axis*

batch_dims *
Tindices0*
Taxis0*
Tparams0	*
_output_shapes
: 
╘
Zinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Cast/xPackXinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Prod\input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2*
_output_shapes
:*

axis *
T0	*
N
ч
ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseReshapeSparseReshape3input_layer_1/pay_total_bucketized_embedding/Cast_13input_layer_1/pay_total_bucketized_embedding/Cast_2Zinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
╓
jinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseReshape/IdentityIdentity0input_layer_1/pay_total_bucketized_embedding/add*#
_output_shapes
:         *
T0
д
binput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
ю
`input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GreaterEqualGreaterEqualjinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseReshape/Identitybinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GreaterEqual/y*
T0*#
_output_shapes
:         
Ў
Yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/WhereWhere`input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GreaterEqual*
T0
*'
_output_shapes
:         
┤
ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
р
[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/ReshapeReshapeYinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Whereainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:         *
Tshape0
е
cinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ў
^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_1GatherV2ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseReshape[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshapecinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_1/axis*
Tparams0	*

batch_dims *
Tindices0	*'
_output_shapes
:         *
Taxis0
е
cinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
√
^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_2GatherV2jinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseReshape/Identity[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshapecinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_2/axis*
Tparams0*#
_output_shapes
:         *
Tindices0	*

batch_dims *
Taxis0
Є
\input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/IdentityIdentitycinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
п
minput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
■
{input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_1^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/GatherV2_2\input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Identityminput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╨
input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
╙
Бinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
╙
Бinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
о
yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice{input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackБinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Бinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*
Index0*
T0	*
ellipsis_mask *

begin_mask*
new_axis_mask *
shrink_axis_mask*#
_output_shapes
:         
└
pinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/CastCastyinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:         *
Truncate( *

DstT0
╟
rinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUnique}input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*
out_idx0*2
_output_shapes 
:         :         
Ч
Бinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes
: *
dtype0
 
|input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Cinput_layer_1/pay_total_bucketized_embedding/embedding_weights/readrinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/UniqueБinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0*

batch_dims *
Taxis0*'
_output_shapes
:         *
Tparams0*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights
┬
Еinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity|input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:         
о
kinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeanЕinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitytinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1pinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:         *
T0*

Tidx0
┤
cinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"       
М
]input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_1Reshape}input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2cinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_1/shape*'
_output_shapes
:         *
Tshape0*
T0

Д
Yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/ShapeShapekinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
T0*
out_type0
▒
ginput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
│
iinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
│
iinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Э
ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_sliceStridedSliceYinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Shapeginput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice/stackiinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice/stack_1iinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
end_mask *
ellipsis_mask *
new_axis_mask *

begin_mask *
Index0
Э
[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
█
Yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/stackPack[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/stack/0ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/strided_slice*

axis *
_output_shapes
:*
N*
T0
ч
Xinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/TileTile]input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_1Yinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/stack*0
_output_shapes
:                  *

Tmultiples0*
T0

К
^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/zeros_like	ZerosLikekinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╢
Sinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weightsSelectXinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Tile^input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/zeros_likekinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
█
Zinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Cast_1Cast3input_layer_1/pay_total_bucketized_embedding/Cast_2*

DstT0*

SrcT0	*
_output_shapes
:*
Truncate( 
л
ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
к
`input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
╖
[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_1SliceZinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Cast_1ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_1/begin`input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
ю
[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Shape_1ShapeSinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights*
T0*
_output_shapes
:*
out_type0
л
ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_2/beginConst*
valueB:*
_output_shapes
:*
dtype0
│
`input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_2/sizeConst*
valueB:
         *
_output_shapes
:*
dtype0
╕
[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_2Slice[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Shape_1ainput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_2/begin`input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_2/size*
Index0*
_output_shapes
:*
T0
б
_input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
╗
Zinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/concatConcatV2[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_1[input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Slice_2_input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/concat/axis*

Tidx0*
T0*
_output_shapes
:*
N
┘
]input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_2ReshapeSinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weightsZinput_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/concat*'
_output_shapes
:         *
T0*
Tshape0
╤
4input_layer_1/pay_total_bucketized_embedding/Shape_1Shape]input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_2*
T0*
_output_shapes
:*
out_type0
М
Binput_layer_1/pay_total_bucketized_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
О
Dinput_layer_1/pay_total_bucketized_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
О
Dinput_layer_1/pay_total_bucketized_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
ф
<input_layer_1/pay_total_bucketized_embedding/strided_slice_1StridedSlice4input_layer_1/pay_total_bucketized_embedding/Shape_1Binput_layer_1/pay_total_bucketized_embedding/strided_slice_1/stackDinput_layer_1/pay_total_bucketized_embedding/strided_slice_1/stack_1Dinput_layer_1/pay_total_bucketized_embedding/strided_slice_1/stack_2*
Index0*
new_axis_mask *

begin_mask *
end_mask *
ellipsis_mask *
shrink_axis_mask*
T0*
_output_shapes
: 
А
>input_layer_1/pay_total_bucketized_embedding/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
№
<input_layer_1/pay_total_bucketized_embedding/Reshape_2/shapePack<input_layer_1/pay_total_bucketized_embedding/strided_slice_1>input_layer_1/pay_total_bucketized_embedding/Reshape_2/shape/1*
_output_shapes
:*

axis *
T0*
N
Ю
6input_layer_1/pay_total_bucketized_embedding/Reshape_2Reshape]input_layer_1/pay_total_bucketized_embedding/pay_total_bucketized_embedding_weights/Reshape_2<input_layer_1/pay_total_bucketized_embedding/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
Д
9input_layer_1/show_7d_bucketized_embedding/ExpandDims/dimConst*
_output_shapes
: *
valueB :
         *
dtype0
ц
5input_layer_1/show_7d_bucketized_embedding/ExpandDims
ExpandDims(ParseSingleExample/ParseSingleExample_139input_layer_1/show_7d_bucketized_embedding/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:         
┐
/input_layer_1/show_7d_bucketized_embedding/CastCast5input_layer_1/show_7d_bucketized_embedding/ExpandDims*'
_output_shapes
:         *

SrcT0	*

DstT0*
Truncate( 
М
4input_layer_1/show_7d_bucketized_embedding/Bucketize	Bucketize/input_layer_1/show_7d_bucketized_embedding/Cast*
T0*'
_output_shapes
:         *f

boundariesX
V"T     \F >ГF ▐╦F яG ╧HGАяВGА┐дG 7╩GАоўG└яH@u0H└ MH└qHрЯМH`ъеHАc╔H`хєHЁ╧IЁ ]IЁпJ
д
0input_layer_1/show_7d_bucketized_embedding/ShapeShape4input_layer_1/show_7d_bucketized_embedding/Bucketize*
T0*
_output_shapes
:*
out_type0
И
>input_layer_1/show_7d_bucketized_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
К
@input_layer_1/show_7d_bucketized_embedding/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
К
@input_layer_1/show_7d_bucketized_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╨
8input_layer_1/show_7d_bucketized_embedding/strided_sliceStridedSlice0input_layer_1/show_7d_bucketized_embedding/Shape>input_layer_1/show_7d_bucketized_embedding/strided_slice/stack@input_layer_1/show_7d_bucketized_embedding/strided_slice/stack_1@input_layer_1/show_7d_bucketized_embedding/strided_slice/stack_2*
ellipsis_mask *
shrink_axis_mask*
T0*
new_axis_mask *
Index0*
_output_shapes
: *
end_mask *

begin_mask 
x
6input_layer_1/show_7d_bucketized_embedding/range/startConst*
value	B : *
_output_shapes
: *
dtype0
x
6input_layer_1/show_7d_bucketized_embedding/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
0input_layer_1/show_7d_bucketized_embedding/rangeRange6input_layer_1/show_7d_bucketized_embedding/range/start8input_layer_1/show_7d_bucketized_embedding/strided_slice6input_layer_1/show_7d_bucketized_embedding/range/delta*

Tidx0*#
_output_shapes
:         
}
;input_layer_1/show_7d_bucketized_embedding/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Є
7input_layer_1/show_7d_bucketized_embedding/ExpandDims_1
ExpandDims0input_layer_1/show_7d_bucketized_embedding/range;input_layer_1/show_7d_bucketized_embedding/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:         
К
9input_layer_1/show_7d_bucketized_embedding/Tile/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
я
/input_layer_1/show_7d_bucketized_embedding/TileTile7input_layer_1/show_7d_bucketized_embedding/ExpandDims_19input_layer_1/show_7d_bucketized_embedding/Tile/multiples*
T0*

Tmultiples0*'
_output_shapes
:         
Л
8input_layer_1/show_7d_bucketized_embedding/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
ф
2input_layer_1/show_7d_bucketized_embedding/ReshapeReshape/input_layer_1/show_7d_bucketized_embedding/Tile8input_layer_1/show_7d_bucketized_embedding/Reshape/shape*#
_output_shapes
:         *
T0*
Tshape0
z
8input_layer_1/show_7d_bucketized_embedding/range_1/startConst*
dtype0*
_output_shapes
: *
value	B : 
z
8input_layer_1/show_7d_bucketized_embedding/range_1/limitConst*
dtype0*
_output_shapes
: *
value	B :
z
8input_layer_1/show_7d_bucketized_embedding/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
С
2input_layer_1/show_7d_bucketized_embedding/range_1Range8input_layer_1/show_7d_bucketized_embedding/range_1/start8input_layer_1/show_7d_bucketized_embedding/range_1/limit8input_layer_1/show_7d_bucketized_embedding/range_1/delta*

Tidx0*
_output_shapes
:
╖
;input_layer_1/show_7d_bucketized_embedding/Tile_1/multiplesPack8input_layer_1/show_7d_bucketized_embedding/strided_slice*
T0*
N*

axis *
_output_shapes
:
ъ
1input_layer_1/show_7d_bucketized_embedding/Tile_1Tile2input_layer_1/show_7d_bucketized_embedding/range_1;input_layer_1/show_7d_bucketized_embedding/Tile_1/multiples*

Tmultiples0*#
_output_shapes
:         *
T0
Н
:input_layer_1/show_7d_bucketized_embedding/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         
э
4input_layer_1/show_7d_bucketized_embedding/Reshape_1Reshape4input_layer_1/show_7d_bucketized_embedding/Bucketize:input_layer_1/show_7d_bucketized_embedding/Reshape_1/shape*
Tshape0*
T0*#
_output_shapes
:         
r
0input_layer_1/show_7d_bucketized_embedding/mul/xConst*
value	B :*
_output_shapes
: *
dtype0
╚
.input_layer_1/show_7d_bucketized_embedding/mulMul0input_layer_1/show_7d_bucketized_embedding/mul/x1input_layer_1/show_7d_bucketized_embedding/Tile_1*#
_output_shapes
:         *
T0
╦
.input_layer_1/show_7d_bucketized_embedding/addAddV24input_layer_1/show_7d_bucketized_embedding/Reshape_1.input_layer_1/show_7d_bucketized_embedding/mul*#
_output_shapes
:         *
T0
ц
0input_layer_1/show_7d_bucketized_embedding/stackPack2input_layer_1/show_7d_bucketized_embedding/Reshape1input_layer_1/show_7d_bucketized_embedding/Tile_1*'
_output_shapes
:         *
T0*

axis *
N
К
9input_layer_1/show_7d_bucketized_embedding/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
э
4input_layer_1/show_7d_bucketized_embedding/transpose	Transpose0input_layer_1/show_7d_bucketized_embedding/stack9input_layer_1/show_7d_bucketized_embedding/transpose/perm*'
_output_shapes
:         *
T0*
Tperm0
└
1input_layer_1/show_7d_bucketized_embedding/Cast_1Cast4input_layer_1/show_7d_bucketized_embedding/transpose*
Truncate( *

SrcT0*'
_output_shapes
:         *

DstT0	
v
4input_layer_1/show_7d_bucketized_embedding/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
ф
2input_layer_1/show_7d_bucketized_embedding/stack_1Pack8input_layer_1/show_7d_bucketized_embedding/strided_slice4input_layer_1/show_7d_bucketized_embedding/stack_1/1*
N*
_output_shapes
:*

axis *
T0
▒
1input_layer_1/show_7d_bucketized_embedding/Cast_2Cast2input_layer_1/show_7d_bucketized_embedding/stack_1*
Truncate( *
_output_shapes
:*

SrcT0*

DstT0	
Б
_input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
valueB"      
Ї
^input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/meanConst*
dtype0*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
_output_shapes
: *
valueB
 *    
Ў
`input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *   ?*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights
¤
iinput_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal_input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/shape*
T0*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
dtype0*
seed2 *

seed *
_output_shapes

:
л
]input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mulMuliinput_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/TruncatedNormal`input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights
Щ
Yinput_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normalAdd]input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mul^input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal/mean*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
T0*
_output_shapes

:
Б
<input_layer_1/show_7d_bucketized_embedding/embedding_weights
VariableV2*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
dtype0*
shape
:*
_output_shapes

:*
shared_name *
	container 
Й
Cinput_layer_1/show_7d_bucketized_embedding/embedding_weights/AssignAssign<input_layer_1/show_7d_bucketized_embedding/embedding_weightsYinput_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
T0*
use_locking(*
validate_shape(*
_output_shapes

:
Е
Ainput_layer_1/show_7d_bucketized_embedding/embedding_weights/readIdentity<input_layer_1/show_7d_bucketized_embedding/embedding_weights*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
T0*
_output_shapes

:
е
[input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice/beginConst*
dtype0*
valueB: *
_output_shapes
:
д
Zinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
№
Uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SliceSlice1input_layer_1/show_7d_bucketized_embedding/Cast_2[input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice/beginZinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice/size*
Index0*
_output_shapes
:*
T0	
Я
Uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/ConstConst*
dtype0*
valueB: *
_output_shapes
:
╚
Tinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/ProdProdUinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SliceUinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Const*
	keep_dims( *
_output_shapes
: *
T0	*

Tidx0
в
`input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2/indicesConst*
value	B :*
_output_shapes
: *
dtype0
Я
]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
о
Xinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2GatherV21input_layer_1/show_7d_bucketized_embedding/Cast_2`input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2/indices]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2/axis*
Taxis0*
Tparams0	*
Tindices0*

batch_dims *
_output_shapes
: 
╚
Vinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Cast/xPackTinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/ProdXinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2*
_output_shapes
:*
N*
T0	*

axis 
█
]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseReshapeSparseReshape1input_layer_1/show_7d_bucketized_embedding/Cast_11input_layer_1/show_7d_bucketized_embedding/Cast_2Vinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Cast/x*-
_output_shapes
:         :
╨
finput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseReshape/IdentityIdentity.input_layer_1/show_7d_bucketized_embedding/add*#
_output_shapes
:         *
T0
а
^input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
т
\input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GreaterEqualGreaterEqualfinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseReshape/Identity^input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GreaterEqual/y*#
_output_shapes
:         *
T0
ю
Uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/WhereWhere\input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GreaterEqual*'
_output_shapes
:         *
T0

░
]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╘
Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/ReshapeReshapeUinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Where]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape/shape*
Tshape0*
T0	*#
_output_shapes
:         
б
_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
ц
Zinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_1GatherV2]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseReshapeWinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_1/axis*'
_output_shapes
:         *
Taxis0*
Tparams0	*

batch_dims *
Tindices0	
б
_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
ы
Zinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_2GatherV2finput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseReshape/IdentityWinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_2/axis*#
_output_shapes
:         *
Taxis0*
Tparams0*
Tindices0	*

batch_dims 
ъ
Xinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/IdentityIdentity_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
л
iinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B : *
_output_shapes
: *
dtype0
ъ
winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsZinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_1Zinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/GatherV2_2Xinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Identityiinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:         :         :         :         *
T0
╠
{input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
╬
}input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
╬
}input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ш
uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicewinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows{input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack}input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1}input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*#
_output_shapes
:         *
ellipsis_mask *
Index0*
end_mask*
new_axis_mask *

begin_mask*
shrink_axis_mask
╕
linput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/CastCastuinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*
Truncate( *#
_output_shapes
:         *

DstT0
┐
ninput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/UniqueUniqueyinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
out_idx0*
T0*2
_output_shapes 
:         :         
Р
}input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*
value	B : *
dtype0*
_output_shapes
: *O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights
ю
xinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Ainput_layer_1/show_7d_bucketized_embedding/embedding_weights/readninput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique}input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
Taxis0*

batch_dims *
Tparams0*'
_output_shapes
:         
║
Бinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityxinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:         *
T0
Ю
ginput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparseSparseSegmentMeanБinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identitypinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/Unique:1linput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse/Cast*'
_output_shapes
:         *

Tidx0*
T0
░
_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
А
Yinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_1Reshapeyinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2_input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:         *
Tshape0
№
Uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/ShapeShapeginput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse*
_output_shapes
:*
out_type0*
T0
н
cinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
п
einput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
п
einput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Й
]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_sliceStridedSliceUinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Shapecinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice/stackeinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice/stack_1einput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice/stack_2*
ellipsis_mask *
shrink_axis_mask*
new_axis_mask *
T0*
Index0*

begin_mask *
end_mask *
_output_shapes
: 
Щ
Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
╧
Uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/stackPackWinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/stack/0]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/strided_slice*
_output_shapes
:*
T0*
N*

axis 
█
Tinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/TileTileYinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_1Uinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
В
Zinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/zeros_like	ZerosLikeginput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:         *
T0
ж
Oinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weightsSelectTinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/TileZinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/zeros_likeginput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
╒
Vinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Cast_1Cast1input_layer_1/show_7d_bucketized_embedding/Cast_2*
Truncate( *

SrcT0	*
_output_shapes
:*

DstT0
з
]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
ж
\input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
з
Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_1SliceVinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Cast_1]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_1/begin\input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_1/size*
Index0*
_output_shapes
:*
T0
ц
Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Shape_1ShapeOinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights*
out_type0*
T0*
_output_shapes
:
з
]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
п
\input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
и
Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_2SliceWinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Shape_1]input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_2/begin\input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
Э
[input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
л
Vinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/concatConcatV2Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_1Winput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Slice_2[input_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/concat/axis*

Tidx0*
_output_shapes
:*
T0*
N
═
Yinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_2ReshapeOinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weightsVinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/concat*
Tshape0*
T0*'
_output_shapes
:         
╦
2input_layer_1/show_7d_bucketized_embedding/Shape_1ShapeYinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_2*
out_type0*
T0*
_output_shapes
:
К
@input_layer_1/show_7d_bucketized_embedding/strided_slice_1/stackConst*
valueB: *
_output_shapes
:*
dtype0
М
Binput_layer_1/show_7d_bucketized_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
М
Binput_layer_1/show_7d_bucketized_embedding/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┌
:input_layer_1/show_7d_bucketized_embedding/strided_slice_1StridedSlice2input_layer_1/show_7d_bucketized_embedding/Shape_1@input_layer_1/show_7d_bucketized_embedding/strided_slice_1/stackBinput_layer_1/show_7d_bucketized_embedding/strided_slice_1/stack_1Binput_layer_1/show_7d_bucketized_embedding/strided_slice_1/stack_2*
shrink_axis_mask*
_output_shapes
: *

begin_mask *
new_axis_mask *
Index0*
end_mask *
ellipsis_mask *
T0
~
<input_layer_1/show_7d_bucketized_embedding/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Ў
:input_layer_1/show_7d_bucketized_embedding/Reshape_2/shapePack:input_layer_1/show_7d_bucketized_embedding/strided_slice_1<input_layer_1/show_7d_bucketized_embedding/Reshape_2/shape/1*
T0*

axis *
N*
_output_shapes
:
Ц
4input_layer_1/show_7d_bucketized_embedding/Reshape_2ReshapeYinput_layer_1/show_7d_bucketized_embedding/show_7d_bucketized_embedding_weights/Reshape_2:input_layer_1/show_7d_bucketized_embedding/Reshape_2/shape*'
_output_shapes
:         *
Tshape0*
T0
[
input_layer_1/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
щ
input_layer_1/concatConcatV24input_layer_1/cart_7d_bucketized_embedding/Reshape_25input_layer_1/click_7d_bucketized_embedding/Reshape_23input_layer_1/ctr_7d_bucketized_embedding/Reshape_23input_layer_1/cvr_7d_bucketized_embedding/Reshape_23input_layer_1/ord_7d_bucketized_embedding/Reshape_26input_layer_1/ord_total_bucketized_embedding/Reshape_23input_layer_1/pay_7d_bucketized_embedding/Reshape_26input_layer_1/pay_total_bucketized_embedding/Reshape_24input_layer_1/show_7d_bucketized_embedding/Reshape_2input_layer_1/concat/axis*
N	*
T0*'
_output_shapes
:         $*

Tidx0
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
Р
concatConcatV2input_layer_1/concatinput_layer/concatconcat/axis*
T0*'
_output_shapes
:         H*
N*

Tidx0
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@dense/kernel*
dtype0*
valueB"H      
С
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: *
valueB
 *
╛
С
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *
>*
dtype0*
_output_shapes
: 
ц
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	HА
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
с
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	HА*
_class
loc:@dense/kernel
╙
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	HА
г
dense/kernel
VariableV2*
dtype0*
	container *
_class
loc:@dense/kernel*
shape:	HА*
shared_name *
_output_shapes
:	HА
╚
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
_output_shapes
:	HА*
T0*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
T0*
_output_shapes
:	HА*
_class
loc:@dense/kernel
К
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ч

dense/bias
VariableV2*
shared_name *
_output_shapes	
:А*
shape:А*
	container *
dtype0*
_class
loc:@dense/bias
│
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
_class
loc:@dense/bias*
use_locking(*
T0
l
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes	
:А
К
dense/MatMulMatMulconcatdense/kernel/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b( *
T0
Б
dense/BiasAddBiasAdddense/MatMuldense/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:         А*
T0
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*!
_class
loc:@dense_1/kernel*
valueB"   @   
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *М7╛*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *М7>*
_output_shapes
: *!
_class
loc:@dense_1/kernel
ь
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *
T0*
_output_shapes
:	А@*
dtype0*

seed *!
_class
loc:@dense_1/kernel
╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
щ
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А@
█
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
:	А@*
T0*!
_class
loc:@dense_1/kernel
з
dense_1/kernel
VariableV2*
shared_name *
	container *
_output_shapes
:	А@*!
_class
loc:@dense_1/kernel*
dtype0*
shape:	А@
╨
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
_output_shapes
:	А@*
use_locking(*
T0*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
:	А@
М
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Щ
dense_1/bias
VariableV2*
_output_shapes
:@*
dtype0*
_class
loc:@dense_1/bias*
shared_name *
	container *
shape:@
║
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:@*
validate_shape(
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_output_shapes
:@*
_class
loc:@dense_1/bias
С
dense_1/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_a( *
T0*'
_output_shapes
:         @*
transpose_b( 
Ж
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:         @*
data_formatNHWC*
T0
W
dense_1/ReluReludense_1/BiasAdd*'
_output_shapes
:         @*
T0
г
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
valueB"@       *
dtype0*
_output_shapes
:
Х
-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *  А╛
Х
-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
valueB
 *  А>*
dtype0
ы
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed2 *
_output_shapes

:@ *

seed *!
_class
loc:@dense_2/kernel*
dtype0*
T0
╓
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
ш
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@ *
T0*!
_class
loc:@dense_2/kernel
┌
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes

:@ *
T0
е
dense_2/kernel
VariableV2*
	container *
_output_shapes

:@ *
shape
:@ *!
_class
loc:@dense_2/kernel*
dtype0*
shared_name 
╧
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
_output_shapes

:@ *
use_locking(*
T0*
validate_shape(*!
_class
loc:@dense_2/kernel
{
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes

:@ 
М
dense_2/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: *
valueB *    
Щ
dense_2/bias
VariableV2*
shared_name *
_output_shapes
: *
shape: *
dtype0*
_class
loc:@dense_2/bias*
	container 
║
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense_2/bias*
_output_shapes
: 
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
T0*
_output_shapes
: 
У
dense_2/MatMulMatMuldense_1/Reludense_2/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:          *
T0
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:          
W
dense_2/ReluReludense_2/BiasAdd*'
_output_shapes
:          *
T0
г
/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"       *!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:
Х
-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
valueB
 *JQ┌╛*
_output_shapes
: *
dtype0
Х
-dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
valueB
 *JQ┌>
ы
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes

: *
T0
╓
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
ш
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes

: *
T0*!
_class
loc:@dense_3/kernel
┌
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

: *!
_class
loc:@dense_3/kernel
е
dense_3/kernel
VariableV2*
shared_name *
shape
: *
_output_shapes

: *
	container *!
_class
loc:@dense_3/kernel*
dtype0
╧
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*!
_class
loc:@dense_3/kernel
{
dense_3/kernel/readIdentitydense_3/kernel*
_output_shapes

: *
T0*!
_class
loc:@dense_3/kernel
М
dense_3/bias/Initializer/zerosConst*
_class
loc:@dense_3/bias*
_output_shapes
:*
dtype0*
valueB*    
Щ
dense_3/bias
VariableV2*
dtype0*
shared_name *
shape:*
_output_shapes
:*
_class
loc:@dense_3/bias*
	container 
║
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias
У
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*'
_output_shapes
:         *
transpose_a( *
T0*
transpose_b( 
Ж
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:         
`
head/logits/ShapeShapedense_3/BiasAdd*
T0*
out_type0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
dtype0*
_output_shapes
: *
value	B :
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
g
head/predictions/logisticSigmoiddense_3/BiasAdd*
T0*'
_output_shapes
:         
k
head/predictions/zeros_like	ZerosLikedense_3/BiasAdd*'
_output_shapes
:         *
T0
q
&head/predictions/two_class_logits/axisConst*
dtype0*
_output_shapes
: *
valueB :
         
╩
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likedense_3/BiasAdd&head/predictions/two_class_logits/axis*'
_output_shapes
:         *

Tidx0*
T0*
N
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*
T0*'
_output_shapes
:         
o
$head/predictions/class_ids/dimensionConst*
valueB :
         *
_output_shapes
: *
dtype0
║
head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:         *

Tidx0*
T0
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         
д
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:         *

Tdim0
╒
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*

fill *

scientific( *'
_output_shapes
:         *
	precision         *
shortest( *
width         
e
head/predictions/ShapeShapedense_3/BiasAdd*
out_type0*
_output_shapes
:*
T0
n
$head/predictions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
p
&head/predictions/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
╬
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
ellipsis_mask *
Index0*
end_mask *
T0*

begin_mask *
shrink_axis_mask*
new_axis_mask *
_output_shapes
: 
^
head/predictions/range/startConst*
value	B : *
_output_shapes
: *
dtype0
^
head/predictions/range/limitConst*
dtype0*
_output_shapes
: *
value	B :
^
head/predictions/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
б
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:*

Tidx0
c
!head/predictions/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0
c
!head/predictions/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
д
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
T0*
_output_shapes
:*

axis *
N
б
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:         
g
head/predictions/Shape_1Shapedense_3/BiasAdd*
_output_shapes
:*
out_type0*
T0
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
r
(head/predictions/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
╪
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *

begin_mask *
Index0*
T0*
shrink_axis_mask*
new_axis_mask *
ellipsis_mask 
`
head/predictions/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
value	B :*
dtype0
`
head/predictions/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
й
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:*

Tidx0
┬
head/predictions/AsStringAsStringhead/predictions/range_1*
shortest( *

scientific( *

fill *
width         *
T0*
_output_shapes
:*
	precision         
c
!head/predictions/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ю
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*

Tdim0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
к
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*

axis *
N*
T0*
_output_shapes
:
е
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*

Tmultiples0*'
_output_shapes
:         
h

head/ShapeShapehead/predictions/probabilities*
out_type0*
_output_shapes
:*
T0
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
d
head/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Т
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
shrink_axis_mask*
T0*
ellipsis_mask *

begin_mask 
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
dtype0*
value	B :*
_output_shapes
: 
R
head/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
q

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:*

Tidx0
и
head/AsStringAsString
head/range*
T0*
	precision         *
_output_shapes
:*
width         *

fill *

scientific( *
shortest( 
U
head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
v
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
_output_shapes

:*

Tdim0*
T0
W
head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
А
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
_output_shapes
:*
N*
T0*

axis 
{
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:         *

Tmultiples0

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
l
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const*
pattern
^s3://.**
_output_shapes
: 
s
save/cond/SwitchSwitchsave/StaticRegexFullMatchsave/StaticRegexFullMatch*
_output_shapes
: : *
T0

S
save/cond/switch_tIdentitysave/cond/Switch:1*
T0
*
_output_shapes
: 
Q
save/cond/switch_fIdentitysave/cond/Switch*
_output_shapes
: *
T0

Y
save/cond/pred_idIdentitysave/StaticRegexFullMatch*
_output_shapes
: *
T0

j
save/cond/ConstConst^save/cond/switch_t*
valueB B.part*
_output_shapes
: *
dtype0
Т
save/cond/Const_1Const^save/cond/switch_f*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_03dfe2aff9894dfa85a57ecae80d64d8/part
h
save/cond/MergeMergesave/cond/Const_1save/cond/Const*
T0*
_output_shapes
: : *
N
l
save/StringJoin
StringJoin
save/Constsave/cond/Merge*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
▓
save/SaveV2/tensor_namesConst"/device:CPU:0*╓
value╠B╔B
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBglobal_stepB6input_layer/cate_level1_id_embedding/embedding_weightsB6input_layer/cate_level2_id_embedding/embedding_weightsB6input_layer/cate_level3_id_embedding/embedding_weightsB6input_layer/cate_level4_id_embedding/embedding_weightsB/input_layer/country_embedding/embedding_weightsB<input_layer_1/cart_7d_bucketized_embedding/embedding_weightsB=input_layer_1/click_7d_bucketized_embedding/embedding_weightsB;input_layer_1/ctr_7d_bucketized_embedding/embedding_weightsB;input_layer_1/cvr_7d_bucketized_embedding/embedding_weightsB;input_layer_1/ord_7d_bucketized_embedding/embedding_weightsB>input_layer_1/ord_total_bucketized_embedding/embedding_weightsB;input_layer_1/pay_7d_bucketized_embedding/embedding_weightsB>input_layer_1/pay_total_bucketized_embedding/embedding_weightsB<input_layer_1/show_7d_bucketized_embedding/embedding_weights*
_output_shapes
:*
dtype0
а
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
┌
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices
dense/biasdense/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kernelglobal_step6input_layer/cate_level1_id_embedding/embedding_weights6input_layer/cate_level2_id_embedding/embedding_weights6input_layer/cate_level3_id_embedding/embedding_weights6input_layer/cate_level4_id_embedding/embedding_weights/input_layer/country_embedding/embedding_weights<input_layer_1/cart_7d_bucketized_embedding/embedding_weights=input_layer_1/click_7d_bucketized_embedding/embedding_weights;input_layer_1/ctr_7d_bucketized_embedding/embedding_weights;input_layer_1/cvr_7d_bucketized_embedding/embedding_weights;input_layer_1/ord_7d_bucketized_embedding/embedding_weights>input_layer_1/ord_total_bucketized_embedding/embedding_weights;input_layer_1/pay_7d_bucketized_embedding/embedding_weights>input_layer_1/pay_total_bucketized_embedding/embedding_weights<input_layer_1/show_7d_bucketized_embedding/embedding_weights"/device:CPU:0*%
dtypes
2	
а
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
м
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
_output_shapes
:*
N
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
╡
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╓
value╠B╔B
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBglobal_stepB6input_layer/cate_level1_id_embedding/embedding_weightsB6input_layer/cate_level2_id_embedding/embedding_weightsB6input_layer/cate_level3_id_embedding/embedding_weightsB6input_layer/cate_level4_id_embedding/embedding_weightsB/input_layer/country_embedding/embedding_weightsB<input_layer_1/cart_7d_bucketized_embedding/embedding_weightsB=input_layer_1/click_7d_bucketized_embedding/embedding_weightsB;input_layer_1/ctr_7d_bucketized_embedding/embedding_weightsB;input_layer_1/cvr_7d_bucketized_embedding/embedding_weightsB;input_layer_1/ord_7d_bucketized_embedding/embedding_weightsB>input_layer_1/ord_total_bucketized_embedding/embedding_weightsB;input_layer_1/pay_7d_bucketized_embedding/embedding_weightsB>input_layer_1/pay_total_bucketized_embedding/embedding_weightsB<input_layer_1/show_7d_bucketized_embedding/embedding_weights
г
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
Н
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	
Я
save/AssignAssign
dense/biassave/RestoreV2*
_output_shapes	
:А*
T0*
validate_shape(*
_class
loc:@dense/bias*
use_locking(
л
save/Assign_1Assigndense/kernelsave/RestoreV2:1*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(*
T0*
_output_shapes
:	HА
ж
save/Assign_2Assigndense_1/biassave/RestoreV2:2*
_output_shapes
:@*
validate_shape(*
_class
loc:@dense_1/bias*
T0*
use_locking(
п
save/Assign_3Assigndense_1/kernelsave/RestoreV2:3*!
_class
loc:@dense_1/kernel*
use_locking(*
_output_shapes
:	А@*
T0*
validate_shape(
ж
save/Assign_4Assigndense_2/biassave/RestoreV2:4*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@dense_2/bias
о
save/Assign_5Assigndense_2/kernelsave/RestoreV2:5*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel*
_output_shapes

:@ 
ж
save/Assign_6Assigndense_3/biassave/RestoreV2:6*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
о
save/Assign_7Assigndense_3/kernelsave/RestoreV2:7*
use_locking(*
_output_shapes

: *
validate_shape(*!
_class
loc:@dense_3/kernel*
T0
а
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
use_locking(*
validate_shape(*
T0	*
_output_shapes
: *
_class
loc:@global_step
■
save/Assign_9Assign6input_layer/cate_level1_id_embedding/embedding_weightssave/RestoreV2:9*
validate_shape(*
_output_shapes

:d*I
_class?
=;loc:@input_layer/cate_level1_id_embedding/embedding_weights*
T0*
use_locking(
Б
save/Assign_10Assign6input_layer/cate_level2_id_embedding/embedding_weightssave/RestoreV2:10*
validate_shape(*
_output_shapes
:	Р*I
_class?
=;loc:@input_layer/cate_level2_id_embedding/embedding_weights*
T0*
use_locking(
Б
save/Assign_11Assign6input_layer/cate_level3_id_embedding/embedding_weightssave/RestoreV2:11*
T0*
_output_shapes
:	ш*
validate_shape(*I
_class?
=;loc:@input_layer/cate_level3_id_embedding/embedding_weights*
use_locking(
Б
save/Assign_12Assign6input_layer/cate_level4_id_embedding/embedding_weightssave/RestoreV2:12*I
_class?
=;loc:@input_layer/cate_level4_id_embedding/embedding_weights*
T0*
_output_shapes
:	╨*
use_locking(*
validate_shape(
Є
save/Assign_13Assign/input_layer/country_embedding/embedding_weightssave/RestoreV2:13*
use_locking(*B
_class8
64loc:@input_layer/country_embedding/embedding_weights*
validate_shape(*
_output_shapes

:*
T0
М
save/Assign_14Assign<input_layer_1/cart_7d_bucketized_embedding/embedding_weightssave/RestoreV2:14*
T0*
validate_shape(*
use_locking(*O
_classE
CAloc:@input_layer_1/cart_7d_bucketized_embedding/embedding_weights*
_output_shapes

:
О
save/Assign_15Assign=input_layer_1/click_7d_bucketized_embedding/embedding_weightssave/RestoreV2:15*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*P
_classF
DBloc:@input_layer_1/click_7d_bucketized_embedding/embedding_weights
К
save/Assign_16Assign;input_layer_1/ctr_7d_bucketized_embedding/embedding_weightssave/RestoreV2:16*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*N
_classD
B@loc:@input_layer_1/ctr_7d_bucketized_embedding/embedding_weights
К
save/Assign_17Assign;input_layer_1/cvr_7d_bucketized_embedding/embedding_weightssave/RestoreV2:17*
validate_shape(*
T0*
_output_shapes

:*
use_locking(*N
_classD
B@loc:@input_layer_1/cvr_7d_bucketized_embedding/embedding_weights
К
save/Assign_18Assign;input_layer_1/ord_7d_bucketized_embedding/embedding_weightssave/RestoreV2:18*
validate_shape(*
T0*
_output_shapes

:*N
_classD
B@loc:@input_layer_1/ord_7d_bucketized_embedding/embedding_weights*
use_locking(
Р
save/Assign_19Assign>input_layer_1/ord_total_bucketized_embedding/embedding_weightssave/RestoreV2:19*
validate_shape(*Q
_classG
ECloc:@input_layer_1/ord_total_bucketized_embedding/embedding_weights*
_output_shapes

:*
T0*
use_locking(
К
save/Assign_20Assign;input_layer_1/pay_7d_bucketized_embedding/embedding_weightssave/RestoreV2:20*
_output_shapes

:*
use_locking(*N
_classD
B@loc:@input_layer_1/pay_7d_bucketized_embedding/embedding_weights*
T0*
validate_shape(
Р
save/Assign_21Assign>input_layer_1/pay_total_bucketized_embedding/embedding_weightssave/RestoreV2:21*Q
_classG
ECloc:@input_layer_1/pay_total_bucketized_embedding/embedding_weights*
_output_shapes

:*
use_locking(*
T0*
validate_shape(
М
save/Assign_22Assign<input_layer_1/show_7d_bucketized_embedding/embedding_weightssave/RestoreV2:22*
use_locking(*
_output_shapes

:*
validate_shape(*O
_classE
CAloc:@input_layer_1/show_7d_bucketized_embedding/embedding_weights*
T0
Х
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"Ж<
save/Const:0save/Identity:0save/restore_all (5 @F8"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"я
model_variables█╪
С
8input_layer/cate_level1_id_embedding/embedding_weights:0=input_layer/cate_level1_id_embedding/embedding_weights/Assign=input_layer/cate_level1_id_embedding/embedding_weights/read:02Uinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level2_id_embedding/embedding_weights:0=input_layer/cate_level2_id_embedding/embedding_weights/Assign=input_layer/cate_level2_id_embedding/embedding_weights/read:02Uinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level3_id_embedding/embedding_weights:0=input_layer/cate_level3_id_embedding/embedding_weights/Assign=input_layer/cate_level3_id_embedding/embedding_weights/read:02Uinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level4_id_embedding/embedding_weights:0=input_layer/cate_level4_id_embedding/embedding_weights/Assign=input_layer/cate_level4_id_embedding/embedding_weights/read:02Uinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal:08
ї
1input_layer/country_embedding/embedding_weights:06input_layer/country_embedding/embedding_weights/Assign6input_layer/country_embedding/embedding_weights/read:02Ninput_layer/country_embedding/embedding_weights/Initializer/truncated_normal:08
й
>input_layer_1/cart_7d_bucketized_embedding/embedding_weights:0Cinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/AssignCinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/read:02[input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
н
?input_layer_1/click_7d_bucketized_embedding/embedding_weights:0Dinput_layer_1/click_7d_bucketized_embedding/embedding_weights/AssignDinput_layer_1/click_7d_bucketized_embedding/embedding_weights/read:02\input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/ctr_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/cvr_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/ord_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/ord_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
▒
@input_layer_1/ord_total_bucketized_embedding/embedding_weights:0Einput_layer_1/ord_total_bucketized_embedding/embedding_weights/AssignEinput_layer_1/ord_total_bucketized_embedding/embedding_weights/read:02]input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/pay_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/pay_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
▒
@input_layer_1/pay_total_bucketized_embedding/embedding_weights:0Einput_layer_1/pay_total_bucketized_embedding/embedding_weights/AssignEinput_layer_1/pay_total_bucketized_embedding/embedding_weights/read:02]input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
й
>input_layer_1/show_7d_bucketized_embedding/embedding_weights:0Cinput_layer_1/show_7d_bucketized_embedding/embedding_weights/AssignCinput_layer_1/show_7d_bucketized_embedding/embedding_weights/read:02[input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08"з&
trainable_variablesП&М&
С
8input_layer/cate_level1_id_embedding/embedding_weights:0=input_layer/cate_level1_id_embedding/embedding_weights/Assign=input_layer/cate_level1_id_embedding/embedding_weights/read:02Uinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level2_id_embedding/embedding_weights:0=input_layer/cate_level2_id_embedding/embedding_weights/Assign=input_layer/cate_level2_id_embedding/embedding_weights/read:02Uinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level3_id_embedding/embedding_weights:0=input_layer/cate_level3_id_embedding/embedding_weights/Assign=input_layer/cate_level3_id_embedding/embedding_weights/read:02Uinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level4_id_embedding/embedding_weights:0=input_layer/cate_level4_id_embedding/embedding_weights/Assign=input_layer/cate_level4_id_embedding/embedding_weights/read:02Uinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal:08
ї
1input_layer/country_embedding/embedding_weights:06input_layer/country_embedding/embedding_weights/Assign6input_layer/country_embedding/embedding_weights/read:02Ninput_layer/country_embedding/embedding_weights/Initializer/truncated_normal:08
й
>input_layer_1/cart_7d_bucketized_embedding/embedding_weights:0Cinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/AssignCinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/read:02[input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
н
?input_layer_1/click_7d_bucketized_embedding/embedding_weights:0Dinput_layer_1/click_7d_bucketized_embedding/embedding_weights/AssignDinput_layer_1/click_7d_bucketized_embedding/embedding_weights/read:02\input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/ctr_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/cvr_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/ord_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/ord_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
▒
@input_layer_1/ord_total_bucketized_embedding/embedding_weights:0Einput_layer_1/ord_total_bucketized_embedding/embedding_weights/AssignEinput_layer_1/ord_total_bucketized_embedding/embedding_weights/read:02]input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/pay_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/pay_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
▒
@input_layer_1/pay_total_bucketized_embedding/embedding_weights:0Einput_layer_1/pay_total_bucketized_embedding/embedding_weights/AssignEinput_layer_1/pay_total_bucketized_embedding/embedding_weights/read:02]input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
й
>input_layer_1/show_7d_bucketized_embedding/embedding_weights:0Cinput_layer_1/show_7d_bucketized_embedding/embedding_weights/AssignCinput_layer_1/show_7d_bucketized_embedding/embedding_weights/read:02[input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
o
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:08
o
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:08"∙&
	variablesы&ш&
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
С
8input_layer/cate_level1_id_embedding/embedding_weights:0=input_layer/cate_level1_id_embedding/embedding_weights/Assign=input_layer/cate_level1_id_embedding/embedding_weights/read:02Uinput_layer/cate_level1_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level2_id_embedding/embedding_weights:0=input_layer/cate_level2_id_embedding/embedding_weights/Assign=input_layer/cate_level2_id_embedding/embedding_weights/read:02Uinput_layer/cate_level2_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level3_id_embedding/embedding_weights:0=input_layer/cate_level3_id_embedding/embedding_weights/Assign=input_layer/cate_level3_id_embedding/embedding_weights/read:02Uinput_layer/cate_level3_id_embedding/embedding_weights/Initializer/truncated_normal:08
С
8input_layer/cate_level4_id_embedding/embedding_weights:0=input_layer/cate_level4_id_embedding/embedding_weights/Assign=input_layer/cate_level4_id_embedding/embedding_weights/read:02Uinput_layer/cate_level4_id_embedding/embedding_weights/Initializer/truncated_normal:08
ї
1input_layer/country_embedding/embedding_weights:06input_layer/country_embedding/embedding_weights/Assign6input_layer/country_embedding/embedding_weights/read:02Ninput_layer/country_embedding/embedding_weights/Initializer/truncated_normal:08
й
>input_layer_1/cart_7d_bucketized_embedding/embedding_weights:0Cinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/AssignCinput_layer_1/cart_7d_bucketized_embedding/embedding_weights/read:02[input_layer_1/cart_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
н
?input_layer_1/click_7d_bucketized_embedding/embedding_weights:0Dinput_layer_1/click_7d_bucketized_embedding/embedding_weights/AssignDinput_layer_1/click_7d_bucketized_embedding/embedding_weights/read:02\input_layer_1/click_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/ctr_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/ctr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/cvr_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/cvr_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/ord_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/ord_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/ord_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
▒
@input_layer_1/ord_total_bucketized_embedding/embedding_weights:0Einput_layer_1/ord_total_bucketized_embedding/embedding_weights/AssignEinput_layer_1/ord_total_bucketized_embedding/embedding_weights/read:02]input_layer_1/ord_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
е
=input_layer_1/pay_7d_bucketized_embedding/embedding_weights:0Binput_layer_1/pay_7d_bucketized_embedding/embedding_weights/AssignBinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/read:02Zinput_layer_1/pay_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
▒
@input_layer_1/pay_total_bucketized_embedding/embedding_weights:0Einput_layer_1/pay_total_bucketized_embedding/embedding_weights/AssignEinput_layer_1/pay_total_bucketized_embedding/embedding_weights/read:02]input_layer_1/pay_total_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
й
>input_layer_1/show_7d_bucketized_embedding/embedding_weights:0Cinput_layer_1/show_7d_bucketized_embedding/embedding_weights/AssignCinput_layer_1/show_7d_bucketized_embedding/embedding_weights/read:02[input_layer_1/show_7d_bucketized_embedding/embedding_weights/Initializer/truncated_normal:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
o
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
o
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:08
^
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:08
o
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:08
^
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:08"%
saved_model_main_op


group_deps"°
cond_contextчф
о
save/cond/cond_textsave/cond/pred_id:0save/cond/switch_t:0 *j
save/cond/Const:0
save/cond/pred_id:0
save/cond/switch_t:0*
save/cond/pred_id:0save/cond/pred_id:0
░
save/cond/cond_text_1save/cond/pred_id:0save/cond/switch_f:0*l
save/cond/Const_1:0
save/cond/pred_id:0
save/cond/switch_f:0*
save/cond/pred_id:0save/cond/pred_id:0*К
predict■
F
ctr_7d<
)ParseSingleExample/ParseSingleExample_7:0         
N
cate_level1_id<
)ParseSingleExample/ParseSingleExample_1:0         
F
cvr_7d<
)ParseSingleExample/ParseSingleExample_8:0         
N
cate_level3_id<
)ParseSingleExample/ParseSingleExample_3:0         
G
pay_7d=
*ParseSingleExample/ParseSingleExample_11:0	         
N
cate_level4_id<
)ParseSingleExample/ParseSingleExample_4:0         
H
show_7d=
*ParseSingleExample/ParseSingleExample_13:0	         
N
cate_level2_id<
)ParseSingleExample/ParseSingleExample_2:0         
J
	pay_total=
*ParseSingleExample/ParseSingleExample_12:0	         
J
	ord_total=
*ParseSingleExample/ParseSingleExample_10:0	         
E
cart_7d:
'ParseSingleExample/ParseSingleExample:0	         
F
ord_7d<
)ParseSingleExample/ParseSingleExample_9:0	         
G
country<
)ParseSingleExample/ParseSingleExample_6:0         
H
click_7d<
)ParseSingleExample/ParseSingleExample_5:0	         A
	class_ids4
head/predictions/ExpandDims:0	         H
probabilities7
 head/predictions/probabilities:0         ?
all_class_ids.
head/predictions/Tile:0         2
logits(
dense_3/BiasAdd:0         >
logistic2
head/predictions/logistic:0         ?
all_classes0
head/predictions/Tile_1:0         @
classes5
head/predictions/str_classes:0         tensorflow/serving/predict