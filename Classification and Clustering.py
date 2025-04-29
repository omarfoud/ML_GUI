import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pickle 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class MLClassificationApp:
    # ---------------------
    # GUI Setup & Initialization
    # ---------------------
    def __init__(self, root):
        self.root = root
        self.root.title("ML Classification and Clustering Tool")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.file_path = None
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title_label = tk.Label(
            self.main_frame, 
            text="ML Classification and Clustering Tool", 
            font=("Helvetica", 18, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=10)
        self.upload_button = tk.Button(
            self.main_frame,
            text="Upload CSV File",
            command=self.upload_file,
            font=("Helvetica", 12),
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.upload_button.pack(pady=10)
        self.file_info_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.file_info_frame.pack(fill=tk.X, pady=5)
        
        self.file_label = tk.Label(
            self.file_info_frame,
            text="No file selected",
            font=("Helvetica", 10),
            bg="#f0f0f0"
        )
        self.file_label.pack(side=tk.LEFT, padx=5)
        self.options_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.options_frame.pack(fill=tk.X, pady=10)
        self.target_label = tk.Label(
            self.options_frame,
            text="Target Column:",
            font=("Helvetica", 10),
            bg="#f0f0f0"
        )
        self.target_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(
            self.options_frame,
            textvariable=self.target_var,
            state="disabled"
        )
        self.target_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.test_size_label = tk.Label(
            self.options_frame,
            text="Test Size (%):",
            font=("Helvetica", 10),
            bg="#f0f0f0"
        )
        self.test_size_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        self.test_size_var = tk.StringVar(value="20")
        self.test_size_entry = tk.Entry(
            self.options_frame,
            textvariable=self.test_size_var,
            width=5
        )
        self.test_size_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        self.classify_button = tk.Button(
            self.options_frame,
            text="Run Classification",
            command=self.run_classification,
            font=("Helvetica", 10),
            bg="#2196F3",
            fg="white",
            state="disabled",
            padx=10,
            pady=3
        )
        self.classify_button.grid(row=0, column=4, padx=20, pady=5, sticky=tk.W)
        self.clusters_label = tk.Label(
            self.options_frame,
            text="Clusters:",
            font=("Helvetica", 10),
            bg="#f0f0f0"
        )
        self.clusters_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.clusters_var = tk.StringVar(value="3")
        self.clusters_entry = tk.Entry(
            self.options_frame,
            textvariable=self.clusters_var,
            width=5
        )
        self.clusters_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.clustering_button = tk.Button(
            self.options_frame,
            text="Run Clustering",
            command=self.run_clustering,
            font=("Helvetica", 10),
            bg="#FF5722",
            fg="white",
            padx=10,
            pady=3
        )
        self.clustering_button.grid(row=1, column=4, padx=20, pady=5, sticky=tk.W)
        self.status_label = tk.Label(
            self.main_frame,
            text="",
            font=("Helvetica", 10),
            bg="#f0f0f0",
            fg="green"
        )
        self.status_label.pack(pady=5)
        self.results_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.initial_message = tk.Label(
            self.results_frame,
            text="Upload a CSV file to begin",
            font=("Helvetica", 12),
            bg="#f0f0f0",
            fg="#555555"
        )
        self.initial_message.pack(pady=50)
    
    # ---------------------
    # File Upload and Processing
    # ---------------------
    def upload_file(self):
        """Function to upload and process CSV file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
                
            self.file_path = file_path
            self.file_label.config(text=f"Selected file: {os.path.basename(file_path)}")
            self.df = pd.read_csv(file_path)
            self.target_dropdown['values'] = list(self.df.columns)
            self.target_dropdown.current(len(self.df.columns) - 1)  
            self.target_dropdown['state'] = 'readonly'
            self.classify_button['state'] = 'normal'
            self.show_data_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    # ---------------------
    # Data Preview Display
    # ---------------------
    def show_data_preview(self):
        """Display preview of the loaded data"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        preview_frame = tk.Frame(self.results_frame, bg="#f0f0f0")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        preview_label = tk.Label(
            preview_frame, 
            text="Data Preview:", 
            font=("Helvetica", 12, "bold"),
            bg="#f0f0f0"
        )
        preview_label.pack(anchor="w", pady=(0, 5))
        columns = list(self.df.columns)
        tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")
        for i, row in self.df.head(10).iterrows():
            values = [str(row[col])[:15] + "..." if len(str(row[col])) > 15 else str(row[col]) for col in columns]
            tree.insert("", "end", values=values)
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        info_text = f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}"
        info_label = tk.Label(
            self.results_frame, 
            text=info_text, 
            font=("Helvetica", 10),
            bg="#f0f0f0"
        )
        info_label.pack(anchor="w", pady=5)
    
    # ---------------------
    # Classification Logic
    # ---------------------
    def run_classification(self):
        """Run ML classification on the uploaded data"""
        try:
            if self.df is None:
                messagebox.showerror("Error", "No data loaded")
                return
            
            self.status_label.config(text="Training classification model, please wait...")
            self.root.update_idletasks()
            
            target_column = self.target_var.get()
            test_size = float(self.test_size_var.get()) / 100
            
            if not target_column:
                messagebox.showerror("Error", "Please select a target column")
                return
            
            if test_size <= 0 or test_size >= 1:
                messagebox.showerror("Error", "Test size must be between 1 and 99")
                return
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if X[col].nunique() <= 10:
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X = X.drop(columns=[col])
                    X = pd.concat([X, dummies], axis=1)
                else:
                    X[col] = pd.factorize(X[col])[0]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            cm = confusion_matrix(self.y_test, y_pred)
            
            self.show_results(accuracy, report, X.columns, cm)
            self.status_label.config(text="Classification complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_label.config(text="")
    
    # ---------------------
    # Clustering Logic
    # ---------------------
    def run_clustering(self):
        """Run KMeans clustering on the uploaded data and visualize the results"""
        try:
            if self.df is None:
                messagebox.showerror("Error", "No data loaded")
                return
            
            self.status_label.config(text="Running clustering, please wait...")
            self.root.update_idletasks()
            n_clusters = int(self.clusters_var.get())
            X = self.df.copy()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if X[col].nunique() <= 10:
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X = X.drop(columns=[col])
                    X = pd.concat([X, dummies], axis=1)
                else:
                    X[col] = pd.factorize(X[col])[0]
            X = X.select_dtypes(include=[np.number])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            self.df['Cluster'] = cluster_labels
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_vis = pca.fit_transform(X_scaled)
            else:
                X_vis = X_scaled
            fig, ax = plt.subplots(figsize=(6,5))
            scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=cluster_labels, cmap='viridis')
            ax.set_title("KMeans Clustering (PCA-reduced)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            plt.tight_layout()
            
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            notebook = ttk.Notebook(self.results_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            clustering_tab = tk.Frame(notebook, bg="#f0f0f0")
            notebook.add(clustering_tab, text="Clustering Results")
            canvas = FigureCanvasTkAgg(fig, master=clustering_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            metrics_frame = tk.Frame(clustering_tab, bg="#f0f0f0")
            metrics_frame.pack(fill=tk.X, pady=5)
            inertia_label = tk.Label(metrics_frame, text=f"Inertia: {kmeans.inertia_:.4f}", font=("Helvetica", 12), bg="#f0f0f0")
            inertia_label.pack(side=tk.LEFT, padx=10)
            iterations_label = tk.Label(metrics_frame, text=f"Iterations: {kmeans.n_iter_}", font=("Helvetica", 12), bg="#f0f0f0")
            iterations_label.pack(side=tk.LEFT, padx=10)
            
            self.status_label.config(text="Clustering complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed: {str(e)}")
            self.status_label.config(text="")
    
    # ---------------------
    # Results Display and Visualization
    # ---------------------
    def show_results(self, accuracy, report, feature_names, cm):
        """Display classification results and visualizations"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        notebook = ttk.Notebook(self.results_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        results_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(results_tab, text="Results")
        
        accuracy_frame = tk.Frame(results_tab, bg="#f0f0f0")
        accuracy_frame.pack(fill=tk.X, pady=10)
        accuracy_label = tk.Label(accuracy_frame, text=f"Model Accuracy: {accuracy:.4f}", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
        accuracy_label.pack(anchor="center")
        
        report_frame = tk.Frame(results_tab, bg="#f0f0f0")
        report_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        report_label = tk.Label(report_frame, text="Classification Report:", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
        report_label.pack(anchor="w", padx=10)
        
        columns = ("Class", "Precision", "Recall", "F1-Score", "Support")
        report_tree = ttk.Treeview(report_frame, columns=columns, show="headings", height=len(report)-1)
        for col in columns:
            report_tree.heading(col, text=col)
            report_tree.column(col, width=100, anchor="center")
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                values = (class_name,
                          f"{metrics['precision']:.4f}",
                          f"{metrics['recall']:.4f}",
                          f"{metrics['f1-score']:.4f}",
                          f"{metrics['support']}")
                report_tree.insert("", "end", values=values)
        report_tree.pack(padx=10, pady=5, fill=tk.X)
        
        if hasattr(self.model, 'feature_importances_'):
            viz_tab = tk.Frame(notebook, bg="#f0f0f0")
            notebook.add(viz_tab, text="Feature Importance")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title("Feature Importance")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            ax.bar(range(len(importances)), importances[indices])
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=viz_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        cm_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(cm_tab, text="Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax_cm.figure.colorbar(im, ax=ax_cm)
        classes = np.unique(self.y_test)
        ax_cm.set(xticks=np.arange(len(classes)),
                  yticks=np.arange(len(classes)),
                  xticklabels=classes, yticklabels=classes,
                  xlabel="Predicted label",
                  ylabel="True label",
                  title="Confusion Matrix")
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_tab)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        export_frame = tk.Frame(self.results_frame, bg="#f0f0f0")
        export_frame.pack(pady=10)
        export_results_button = tk.Button(
            export_frame,
            text="Export Results",
            command=self.export_results,
            font=("Helvetica", 10),
            bg="#FF9800",
            fg="white",
            padx=10,
            pady=3
        )
        export_results_button.pack(side=tk.LEFT, padx=10)
        if self.model is not None:
            export_model_button = tk.Button(
                export_frame,
                text="Export Model",
                command=self.export_model,
                font=("Helvetica", 10),
                bg="#9C27B0",
                fg="white",
                padx=10,
                pady=3
            )
            export_model_button.pack(side=tk.LEFT, padx=10)
    def export_results(self):
        """Export the classification results to a text file"""
        if self.model is None:
            messagebox.showerror("Error", "No model to export results from")
            return
        try:
            export_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not export_path:
                return
            with open(export_path, 'w') as f:
                f.write(f"Classification Results\n")
                f.write(f"====================\n\n")
                f.write(f"Dataset: {os.path.basename(self.file_path)}\n")
                f.write(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}\n\n")
                f.write(f"Model: Random Forest Classifier\n")
                f.write(f"Training set size: {len(self.X_train)}\n")
                f.write(f"Test set size: {len(self.X_test)}\n\n")
                f.write(f"Accuracy: {accuracy_score(self.y_test, self.model.predict(self.X_test)):.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(self.y_test, self.model.predict(self.X_test)))
                if hasattr(self.model, 'feature_importances_'):
                    f.write("\nFeature Importance:\n")
                    importances = self.model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    for i in range(len(importances)):
                        f.write(f"{self.X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}\n")
            messagebox.showinfo("Success", f"Results exported to {export_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    def export_model(self):
        """Export the trained model to a pickle file"""
        if self.model is None:
            messagebox.showerror("Error", "No model to export")
            return
        try:
            model_path = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
            if not model_path:
                return
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            messagebox.showinfo("Success", f"Model exported to {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Model export failed: {str(e)}")
if __name__ == "__main__":
    root = tk.Tk()
    app = MLClassificationApp(root)
    root.mainloop()
