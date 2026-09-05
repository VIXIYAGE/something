import json
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SchemaEx:

    def __init__(self):
        self.feature_order = []

    # =========================================================

    def load_file(self, file_path):
        return pd.read_json(file_path)

    # =========================================================

    def make_hash(self, x):

        if isinstance(x, list):
            return tuple(
                self.make_hash(i)
                for i in x
            )

        elif isinstance(x, dict):
            return json.dumps(
                x,
                sort_keys=True
            )

        return x

    # =========================================================

    def pandint(self, x):

        if isinstance(x, pd.Timestamp):
            return x.isoformat()

        elif hasattr(x, "item"):
            return x.item()

        return x

    # =========================================================

    def detect_and_encode(self, df):

        encoded_columns = []
        self.feature_order = []

        for col in df.columns:

            column_data = df[col]
            dtype = column_data.dtype

            # =================================================
            # INTEGER
            # =================================================

            if pd.api.types.is_integer_dtype(dtype):

                values = column_data.to_numpy(
                    dtype=np.float64
                )

                max_abs = np.max(np.abs(values))
                value_range = np.max(values) - np.min(values)

                if column_data.nunique() <= 1:

                    encoded = np.zeros(len(column_data))

                elif max_abs != 0 and (
                    value_range / max_abs
                ) < 1e-12:

                    encoded = np.zeros(len(column_data))

                else:

                    scaler = StandardScaler()

                    encoded = scaler.fit_transform(
                        values.reshape(-1, 1)
                    ).flatten()

                encoded_columns.append(encoded)
                self.feature_order.append(col)

            # =================================================
            # FLOAT
            # =================================================

            elif pd.api.types.is_float_dtype(dtype):

                values = column_data.to_numpy(
                    dtype=np.float64
                )

                max_abs = np.max(np.abs(values))
                value_range = np.max(values) - np.min(values)

                if column_data.nunique() <= 1:

                    encoded = np.zeros(len(column_data))

                elif max_abs != 0 and (
                    value_range / max_abs
                ) < 1e-12:

                    encoded = np.zeros(len(column_data))

                else:

                    scaler = StandardScaler()

                    encoded = scaler.fit_transform(
                        values.reshape(-1, 1)
                    ).flatten()

                encoded_columns.append(encoded)
                self.feature_order.append(col)

            # =================================================
            # BOOLEAN
            # =================================================

            elif pd.api.types.is_bool_dtype(dtype):

                encoded = (
                    column_data
                    .astype(int)
                    .to_numpy()
                )

                encoded_columns.append(encoded)
                self.feature_order.append(col)

            # =================================================
            # DATETIME
            # =================================================

            elif pd.api.types.is_datetime64_any_dtype(dtype):

                timestamp = (
                    column_data
                    .astype("int64")
                    .to_numpy()
                )

                scaler = StandardScaler()

                encoded = scaler.fit_transform(
                    timestamp.reshape(-1, 1)
                ).flatten()

                encoded_columns.append(encoded)
                self.feature_order.append(col)

            # =================================================
            # STRING / CATEGORICAL / OBJECT
            # =================================================

            elif (
                pd.api.types.is_string_dtype(dtype)
                or pd.api.types.is_object_dtype(dtype)
                or isinstance(
                    dtype,
                    pd.CategoricalDtype
                )
            ):

                values = column_data.apply(
                    self.make_hash
                )

                values = values.astype(str)

                unique_count = values.nunique()

                # LOW CARDINALITY
                if unique_count <= 20:

                    encoder = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore"
                    )

                    encoded = encoder.fit_transform(
                        values
                        .to_numpy()
                        .reshape(-1, 1)
                    )

                    for i in range(
                        encoded.shape[1]
                    ):

                        encoded_columns.append(
                            encoded[:, i]
                        )

                        category_name = (
                            encoder
                            .categories_[0][i]
                        )

                        self.feature_order.append(
                            f"{col}={category_name}"
                        )

                # HIGH CARDINALITY
                else:

                    encoded = (
                        pd.Categorical(values)
                        .codes
                        .astype(np.float32)
                    )

                    encoded_columns.append(
                        encoded
                    )

                    self.feature_order.append(col)

            # =================================================
            # FALLBACK
            # =================================================

            else:

                encoded = (
                    column_data
                    .to_numpy()
                )

                encoded_columns.append(
                    encoded
                )

                self.feature_order.append(col)

        # =====================================================
        # FINAL MATRIX
        # =====================================================

        if len(encoded_columns) == 0:
            return np.empty((len(df), 0))

        encoded_matrix = np.column_stack(
            encoded_columns
        )

        encoded_matrix = np.nan_to_num(
            encoded_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return encoded_matrix

    # =========================================================

    def save_encoded_data(
        self,
        encoded_matrix,
        output_file="data/ecnext2.txt"
    ):

        np.savetxt(
            output_file,
            encoded_matrix,
            fmt="%.6f"
        )

    # =========================================================
    # SAVE SCHEMA
    # =========================================================

    def save_schema(
        self,
        output_file="data/encoder_schema.pkl"
    ):

        joblib.dump(
            self,
            output_file
        )

    # =========================================================
    # LOAD SCHEMA
    # =========================================================

    def load_schema(
        self,
        input_file="data/encoder_schema.pkl"
    ):

        loaded = joblib.load(
            input_file
        )

        self.feature_order = (
            loaded.feature_order
        )

    # =========================================================
    # TRANSFORM
    # =========================================================
    #
    # NOTE:
    # This keeps the same feature structure.
    # For the current UI stage, we mainly use
    # detect_and_encode() so we can test the pipeline.
    #
    # =========================================================

    def transform(self, df):

        encoded_matrix = self.detect_and_encode(df)

        return encoded_matrix


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    extractor = SchemaEx()

    df = extractor.load_file(
        "data/ytr.json"
    )

    df = df.dropna()

    encoded_matrix = (
        extractor.detect_and_encode(df)
    )

    extractor.save_encoded_data(
        encoded_matrix,
        "data/ecnext2.txt"
    )

    extractor.save_schema(
        "data/encoder_schema.pkl"
    )

    print(
        "Encoded shape:",
        encoded_matrix.shape
    )

    print(
        "Feature order:",
        extractor.feature_order
    )

    print(
        "Encoder schema saved."
    )