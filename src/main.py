from Config.Dataset_Config import Kitti
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DETR:
    def __init__(self):
        dataset = Kitti()
        self.dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

        # Load Pretrained DETR Model
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=7, ignore_mismatched_sizes=True)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50") 
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        epochs = 10

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for images, targets in self.dataloader:
                # Convert PIL images to tensor format using DETR processor
                if isinstance(images[0], torch.Tensor):
                    enc_images = torch.stack(images).to(self.device)  # Stack into batch
                else:
                    enc_images = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

                # Fix labels formatting (list of dicts per image)
                labels = []
                for target in targets:
                    labels.append({
                        "class_labels": torch.tensor([t["class_id"] for t in target], dtype=torch.long, device=self.device),
                        "boxes": torch.tensor([t["bbox"] for t in target], dtype=torch.float32, device=self.device),
                    })

                optimizer.zero_grad()
                outputs = self.model(enc_images, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(self.dataloader):.4f}")
            
            
            if epoch%10 == 0:
                torch.save(self.model.state_dict(), f"detr_kitti_finetuned_{epoch//10}.pth")
                
    def visualize_detr_results(self, num_images=1):
        model = self.model
        dataloader= self.dataloader
        device =self.device
        processor = self.processor 
        model.eval()
        with torch.no_grad():
            for idx, (images, targets) in enumerate(dataloader):
                # Convert PIL images to tensor format
                if isinstance(images[0], torch.Tensor):
                    enc_images = torch.stack(images).to(device)  # Stack into batch
                else:
                    enc_images = processor(images=images, return_tensors="pt").pixel_values.to(device)

                outputs = model(enc_images)  # Get the prediction from the model
                logits = outputs.logits
                boxes = outputs.pred_boxes

                for i in range(min(num_images, len(images))):
                    image = images[i]
                    pred_boxes = boxes[i]  # Bounding boxes for the i-th image
                    pred_logits = logits[i]  # Prediction logits for the i-th image

                    # Convert bounding boxes to numpy
                    pred_boxes = pred_boxes.cpu().numpy()
                    scores = pred_logits.softmax(-1)
                    pred_labels = scores.argmax(-1)

                    # Handle image format for displaying
                    if isinstance(image, torch.Tensor):
                        img_width, img_height = image.shape[-2:]  # (H, W) for tensors
                        image = image.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy format
                    else:
                        img_width, img_height = image.size  # (W, H) for PIL Image

                    # Plotting the image
                    fig, ax = plt.subplots(1, figsize=(12, 9))
                    ax.imshow(image)

                    # Ensure bounding boxes are scaled correctly (assuming normalized format)
                    for box, label in zip(pred_boxes, pred_labels):
                        x_center, y_center, width, height = box
                        x_min = (x_center - width / 2) * img_width
                        y_min = (y_center - height / 2) * img_height
                        x_max = (x_center + width / 2) * img_width
                        y_max = (y_center + height / 2) * img_height

                        # Draw bounding box
                        rect = patches.Rectangle(
                            (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)

                        # Add label text
                        ax.text(x_center, y_center, f"Label: {label}", color='white', fontsize=10,
                                bbox=dict(facecolor='red', alpha=0.5))

                    plt.show()

                if idx >= num_images // len(dataloader):
                    break    

if __name__ == "__main__":
    model = DETR()
    print("Model Initialized and Training Started")
    model.train()
    print("Training Ended")
    model.visualize_detr_results()
    