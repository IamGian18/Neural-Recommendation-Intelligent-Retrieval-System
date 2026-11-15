import os
import time
import pickle
import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb


class RetailRocketRecommender:
    """Pipeline de inferencia en dos etapas: FAISS Retrieval + LightGBM Re-Ranking."""

    def __init__(self, artifacts_dir: str = "artifacts", data_dir: str = "data"):
        with open(os.path.join(artifacts_dir, "id_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        self.visitor_to_idx = mappings["visitor_to_idx"]
        self.idx_to_item = mappings["idx_to_item"]

        with open(os.path.join(artifacts_dir, "ltr_meta.pkl"), "rb") as f:
            self.feat_cols = pickle.load(f)["feature_cols"]

        self.user_embs = np.load(os.path.join(artifacts_dir, "user_embeddings.npy")).astype("float32")
        faiss.normalize_L2(self.user_embs)
        self.index = faiss.read_index(os.path.join(artifacts_dir, "faiss_item.index"))
        self.ranker = lgb.Booster(model_file=os.path.join(artifacts_dir, "lgbm_ranker_tuned.txt"))

        n_users_total = len(self.user_embs)
        self.u_lookup = (
            pd.read_parquet(os.path.join(data_dir, "user_features.parquet"))
            .set_index("user_idx")
            .reindex(range(n_users_total))
            .fillna(0)
        )
        self.i_lookup = pd.read_parquet(os.path.join(data_dir, "item_features.parquet")).set_index("item_idx")

        train = pd.read_parquet(os.path.join(data_dir, "train.parquet"), columns=["user_idx", "item_idx", "event", "weight"])
        self.hist_lookup = train.groupby(["user_idx", "item_idx"])["weight"].sum().to_dict()
        self.purchased_lookup = (
            train[train["event"] == "transaction"]
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

    def recommend(self, visitor_id: int, top_k: int = 10, n_candidates: int = 100):
        t0 = time.perf_counter()
        if visitor_id not in self.visitor_to_idx:
            return pd.DataFrame(), 0.0

        u_idx = self.visitor_to_idx[visitor_id]
        excl = self.purchased_lookup.get(u_idx, set())

        # Etapa 1: Recuperación vectorial (FAISS)
        scores, indices = self.index.search(self.user_embs[u_idx : u_idx + 1], n_candidates + 20)
        cands = []
        for item_id, score in zip(indices[0], scores[0]):
            if item_id != -1 and item_id not in excl:
                cands.append((int(item_id), float(score), len(cands)))
                if len(cands) == n_candidates:
                    break

        df_c = pd.DataFrame(cands, columns=["item_idx", "tt_score", "faiss_rank"])
        df_c["user_idx"] = u_idx

        # Etapa 2: Feature engineering en tiempo real
        u_row = self.u_lookup.loc[u_idx]
        for col in self.u_lookup.columns:
            df_c[col] = u_row[col]

        df_c = df_c.merge(self.i_lookup, on="item_idx", how="left").fillna(0)
        df_c["hist_ui_weight"] = [self.hist_lookup.get((u_idx, iid), 0.0) for iid in df_c["item_idx"]]
        df_c["same_fav_cat"] = (df_c["u_fav_cat"] == df_c["categoryid"]).astype(np.float32)
        df_c["pop_x_sim"] = df_c["tt_score"] * np.log1p(df_c["i_weight_sum"])

        # Etapa 3: Re-ranking con LightGBM
        df_c["ltr_score"] = self.ranker.predict(df_c[self.feat_cols])
        top_df = df_c.sort_values("ltr_score", ascending=False).head(top_k).copy()
        top_df["itemid"] = top_df["item_idx"].map(self.idx_to_item)
        top_df["rank"] = range(1, len(top_df) + 1)

        latency_ms = (time.perf_counter() - t0) * 1000
        return top_df[["rank", "itemid", "categoryid", "tt_score", "ltr_score"]], latency_ms


if __name__ == "__main__":
    rec_engine = RetailRocketRecommender()
    sample_id = next(iter(rec_engine.visitor_to_idx.keys()))
    results, elapsed = rec_engine.recommend(sample_id, top_k=10)
    print(f"Visitor ID: {sample_id} | Latencia: {elapsed:.2f} ms")
    print(results.to_string(index=False))
