use crate::primitivelist::bvh::boundingbox::AABB;
use crate::primitivelist::bvh::NodeType::Empty;

pub mod boundingbox;

pub(super) enum NodeType
{
    Empty,
    Leaf
    {
        primitive_index: u32,
    },
    Branch
    {
        split_axis: u8,
        left: Box<BVHNode>,
        right: Box<BVHNode>,
    },
}

impl<'a> Default for NodeType
{
    fn default() -> Self
    {
        Empty
    }
}

#[derive(Default)]
pub(super) struct BVHNode
{
    pub(super) bounding_box: AABB,
    pub(super) node_type: NodeType,
}
