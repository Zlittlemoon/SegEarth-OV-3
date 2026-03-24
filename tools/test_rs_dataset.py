from sam3.train.data.rs_single_query_dataset import RSSingleQueryDataset

ds = RSSingleQueryDataset(
    img_folder="data/DLRSD_split/train/imgs",
    ann_file="data/sam3_semantic_fixed/dlrsd/train.sam3.semantic.json",
    debug=True,
)

x = ds[0]
print(type(x))
print(len(x.images), x.raw_images, len(x.find_queries))
print(x.images[0].data.shape)
print(x.find_queries[0].image_id)
print(x.find_queries[0].query_text)
print(x.find_queries[0].inference_metadata)
print(x.images[0].objects.keys())