# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# import joblib
# import os
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go

# def load_data_and_model(data_dir="."):
#     """Charge les données et le modèle."""
#     try:
#         X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
#         y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
#         model = joblib.load(os.path.join(data_dir, "model.joblib"))
#         return X_test, y_test, model
#     except Exception as e:
#         st.error(f"Erreur lors du chargement des données: {str(e)}")
#         return None, None, None

# def plot_confusion_matrix(y_true, y_pred, model):
#     """Affiche la matrice de confusion avec Plotly."""
#     cm = confusion_matrix(y_true, y_pred)
#     fig = px.imshow(cm,
#                     labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
#                     x=model.classes_,
#                     y=model.classes_,
#                     color_continuous_scale='Blues')
#     fig.update_layout(title='Matrice de Confusion')
#     return fig

# def plot_feature_importance(model, feature_names):
#     """Affiche l'importance des caractéristiques."""
#     if hasattr(model, 'feature_importances_'):
#         importances = pd.DataFrame({
#             'feature': feature_names,
#             'importance': model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         fig = px.bar(importances, 
#                      x='feature', 
#                      y='importance',
#                      title='Importance des Caractéristiques')
#         fig.update_layout(xaxis_tickangle=-45)
#         return fig
#     return None

# def main():
#     st.set_page_config(page_title="Évaluation de Modèle", layout="wide")
    
#     # Titre de l'application
#     st.title("👨‍🔬 Dashboard d'Évaluation de Modèle")
    
#     # Charger les données et le modèle
#     X_test, y_test, model = load_data_and_model()
    
#     if X_test is not None and y_test is not None and model is not None:
#         # Faire des prédictions
#         y_pred = model.predict(X_test)
#         y_test_values = y_test.values.ravel()
        
#         # Créer deux colonnes pour l'affichage
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.subheader("📊 Matrice de Confusion")
#             conf_matrix_fig = plot_confusion_matrix(y_test_values, y_pred, model)
#             st.plotly_chart(conf_matrix_fig, use_container_width=True)
        
#         with col2:
#             st.subheader("📈 Importance des Caractéristiques")
#             feat_imp_fig = plot_feature_importance(model, X_test.columns)
#             if feat_imp_fig is not None:
#                 st.plotly_chart(feat_imp_fig, use_container_width=True)
        
#         # Rapport de classification
#         st.subheader("📝 Rapport de Classification")
#         report = classification_report(y_test_values, y_pred, output_dict=True)
#         df_report = pd.DataFrame(report).transpose()
#         st.dataframe(df_report.style.highlight_max(axis=0), use_container_width=True)
        
#         # Métriques principales
#         st.subheader("🎯 Métriques Principales")
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric(
#                 label="Précision Moyenne",
#                 value=f"{df_report.loc['accuracy', 'precision']:.2%}"
#             )
        
#         with col2:
#             st.metric(
#                 label="Rappel Moyen",
#                 value=f"{df_report.loc['macro avg', 'recall']:.2%}"
#             )
        
#         with col3:
#             st.metric(
#                 label="F1-Score Moyen",
#                 value=f"{df_report.loc['macro avg', 'f1-score']:.2%}"
#             )
        
#         # Visualisation des prédictions
#         st.subheader("🔍 Détails des Prédictions")
#         results_df = pd.DataFrame({
#             'Réalité': y_test_values,
#             'Prédiction': y_pred
#         })
#         st.dataframe(results_df.head(100), use_container_width=True)

# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree, export_text
import joblib
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io

def load_data_and_model(data_dir="."):
    """Charge les données et le modèle."""
    try:
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))
        model = joblib.load(os.path.join(data_dir, "model.joblib"))
        return X_test, y_test, model
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred, model):
    """Affiche la matrice de confusion avec Plotly."""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                    x=model.classes_,
                    y=model.classes_,
                    color_continuous_scale='Blues')
    fig.update_layout(title='Matrice de Confusion')
    return fig
# def plot_confusion_matrix(y_true, y_pred, model):
#     """Affiche la matrice de confusion avec Plotly."""
#     cm = confusion_matrix(y_true, y_pred)
    
#     # On inverse l'ordre des lignes pour avoir la bonne diagonale
#     cm = cm[::-1]
#     classes = model.classes_[::-1]
    
#     # Créer la figure
#     fig = go.Figure(data=go.Heatmap(
#         z=cm,
#         x=model.classes_,  # Classes originales pour l'axe x
#         y=classes,         # Classes inversées pour l'axe y
#         text=cm,
#         texttemplate="%{text}",
#         textfont={"size": 16},
#         colorscale="Blues",
#         showscale=True
#     ))
    
#     # Mise en page
#     fig.update_layout(
#         title='Matrice de Confusion',
#         xaxis_title="Prédiction",
#         yaxis_title="Réalité",
#         xaxis={'side': 'bottom'},
#         width=500,
#         height=500
#     )
    
#     return fig
def plot_feature_importance(model, feature_names):
    """Affiche l'importance des caractéristiques."""
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(importances, 
                     x='feature', 
                     y='importance',
                     title='Importance des Caractéristiques')
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    return None

def plot_decision_tree(model, X_test, max_depth=None):
    """Génère la visualisation de l'arbre de décision."""
    plt.figure(figsize=(20, 10))
    
    if hasattr(model, "estimators_"):
        # Pour Random Forest, on affiche le premier arbre
        tree = model.estimators_[0]
    else:
        # Pour un arbre de décision simple
        tree = model

    plot_tree(
        tree,
        feature_names=X_test.columns,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=max_depth
    )
    
    # Sauvegarder le plot dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    return buf

def get_tree_text(model, X_test, max_depth=None):
    """Obtient la représentation textuelle de l'arbre."""
    if hasattr(model, "estimators_"):
        # Pour Random Forest, on prend le premier arbre
        tree = model.estimators_[0]
    else:
        # Pour un arbre de décision simple
        tree = model
    
    return export_text(
        tree,
        feature_names=list(X_test.columns),
        max_depth=max_depth
    )

def main():
    st.set_page_config(page_title="Évaluation de Modèle", layout="wide")
    
    # Titre de l'application
    st.title("👥 Dashboard d'Évaluation de Modèle")
    
    # Charger les données et le modèle
    X_test, y_test, model = load_data_and_model()
    
    if X_test is not None and y_test is not None and model is not None:
        # Faire des prédictions
        y_pred = model.predict(X_test)
        y_test_values = y_test.values.ravel()
        
        # Navigation par onglets
        tab1, tab2, tab3 = st.tabs(["📊 Métriques", "🌳 Arbre de Décision", "📝 Détails"])
        
        with tab1:
            # Métriques principales
            col1, col2, col3 = st.columns(3)
            report = classification_report(y_test_values, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            
            with col1:
                st.metric(
                    label="Précision Moyenne",
                    value=f"{df_report.loc['accuracy', 'precision']:.2%}"
                )
            with col2:
                st.metric(
                    label="Rappel Moyen",
                    value=f"{df_report.loc['macro avg', 'recall']:.2%}"
                )
            with col3:
                st.metric(
                    label="F1-Score Moyen",
                    value=f"{df_report.loc['macro avg', 'f1-score']:.2%}"
                )
            
            # Visualisations
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Matrice de Confusion")
                conf_matrix_fig = plot_confusion_matrix(y_test_values, y_pred, model)
                st.plotly_chart(conf_matrix_fig, use_container_width=True)
            
            with col2:
                st.subheader("Importance des Caractéristiques")
                feat_imp_fig = plot_feature_importance(model, X_test.columns)
                if feat_imp_fig is not None:
                    st.plotly_chart(feat_imp_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Visualisation de l'Arbre de Décision")
            
            # Options de visualisation
            col1, col2 = st.columns([1, 1])
            with col1:
                max_depth = st.slider(
                    "Profondeur maximale de l'arbre",
                    min_value=1,
                    max_value=10,
                    value=3
                )
            
            with col2:
                view_type = st.radio(
                    "Type de visualisation",
                    ["Graphique", "Texte"]
                )
            
            if view_type == "Graphique":
                tree_buf = plot_decision_tree(model, X_test, max_depth)
                st.image(tree_buf, use_container_width=True)
                
                # Option de téléchargement
                st.download_button(
                    label="📥 Télécharger l'arbre (PNG)",
                    data=tree_buf,
                    file_name="decision_tree.png",
                    mime="image/png"
                )
            else:
                tree_text = get_tree_text(model, X_test, max_depth)
                st.text(tree_text)
                
                # Option de téléchargement
                st.download_button(
                    label="📥 Télécharger l'arbre (TXT)",
                    data=tree_text,
                    file_name="decision_tree.txt",
                    mime="text/plain"
                )
        
        with tab3:
            st.subheader("Rapport de Classification")
            st.dataframe(df_report.style.highlight_max(axis=0), use_container_width=True)
            
            st.subheader("Aperçu des Prédictions")
            results_df = pd.DataFrame({
                'Réalité': y_test_values,
                'Prédiction': y_pred
            })
            st.dataframe(results_df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()