from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd

from notebooks.local.preprocess.pipeline_reader import PreprocBoldRun
from notebooks.local.preprocess.susan import SUSAN


def normalize_column(mc_summary_df: pd.DataFrame, column_label: str, n_start_volumes_to_remove: int) -> np.ndarray:
    return mc_summary_df[column_label][n_start_volumes_to_remove:].values / 10


def normalize_motion_summary_metrics(
    mc_summary_df: pd.DataFrame, n_start_volumes_to_remove: int
) -> Dict[str, np.ndarray]:
    return {key: normalize_column(mc_summary_df, key, n_start_volumes_to_remove) for key in ["framewise_displacement"]}


def min_max_normalize(X):
    _median = np.median(X)
    _min = 0
    _max = X.max()
    _normalized_median = (_median - _min) / (_max - _min)
    return _max, _median, _normalized_median, (X - _min) / (_max - _min)


def denoise_bold(
    bold_run: PreprocBoldRun,
    denoise_settings: Dict[str, Any],
    smooth_mm: Optional[float] = None,
    store_in_tmp: bool = False,
):
    n_start_volumes_to_remove = denoise_settings['n_start_volumes_to_remove']
    # Nuisance regression
    bold_data = bold_run.load_bold(**denoise_settings)

    # Normalize and smooth
    data_type = denoise_settings["data_type"]
    if data_type == "nifti":
        # Get metadata
        raw_data = bold_data["raw"]
        bold_mean = raw_data.mean(-1)
        img = nib.load(bold_run.template_nifti)
        # Add mean back to denoised data
        denoised_data = bold_data["denoised"] + bold_mean[:, :, :, np.newaxis]
        denoised_img = nib.Nifti1Image(denoised_data, affine=img.affine, header=img.header)
        denoised_path = "/tmp/denoised.nii.gz"
        nib.save(denoised_img, denoised_path)
        # Smooth with FSL's susan
        smooth_mm_str = str(smooth_mm).replace(".", "-")
        smoothed_denoised_data = str(bold_run.template_nifti).replace(
            "bold.nii.gz", f"desc-denoised_s-{smooth_mm_str}_bold.nii.gz"
        )

        if store_in_tmp or (smooth_mm is not None and not Path(smoothed_denoised_data).exists()):
            smoothed_denoised_data = (
                f"/tmp/{Path(smoothed_denoised_data).stem}.gz" if store_in_tmp else smoothed_denoised_data
            )
            susan_msg = f"[nifti] Smoothing: {smooth_mm} mm, generating {smoothed_denoised_data}"
            susan_msg = f"[Storing in /tmp] {susan_msg}" if store_in_tmp else susan_msg
            print(susan_msg)
            susan = SUSAN(denoised_path, smoothed_denoised_data, smooth_mm)
            _ = susan.execute()
        else:
            print(f"[nifti] Skip smoothing.\nsmooth: {smooth_mm} mm.\nPath exists: {Path(smoothed_denoised_data)}\n---")

        return None

    elif data_type == "cifti":
        tsnr = bold_data['raw'].mean(1) / bold_data['raw'].std(1)
        tsnr_max, tsnr_median, norm_tsnr_median, tsnr = min_max_normalize(tsnr)
        mc_summary_df = bold_run.load_confounds(
            ["motion_summary_metrics"],
        )

        # plot settings

        """
        # init vertex timeseries plot sample
        vtx_id = 100 # Select vertex index
        vtx_figsize = (6,1)
        vtx_fig, vtx_axs= plt.subplots(2, 1, dpi=200, figsize=vtx_figsize)

        # init carpet plots
        cp_cmap = "Greys_r"
        cp_figsize = (6,2)
        cp_vmin, cp_vmax = -1., 1.
        corr_vmin, corr_vmax = -.08, .08
        cp_fig, cp_axs = plt.subplots(
            3, 2,
            dpi=250, figsize=cp_figsize,
            sharex='col', sharey='row',
            gridspec_kw={
                'height_ratios': [1, 5, 5],
                'width_ratios': [6, 1.4],
            }
        )

        # Plot tsnr
        tsnr_xaxis = list(range(tsnr.shape[0]))
        cp_axs[0,1].text(int(len(tsnr_xaxis)*0), 1.1, f"Median [Max] tSNR: {tsnr_median:.3f} [{tsnr_max:.3f}]", fontsize=FONTSIZE,zorder=10)
        cp_axs[0,1].scatter(tsnr_xaxis, tsnr, marker='s', s=1., c='k', edgecolors='none', alpha=.2)
        cp_axs[0,1].axhline(y=norm_tsnr_median, color='r', zorder=10, linestyle='-', lw=.5)
        """

        # Loop over bold data
        for bold_ix, (desc, _bold_data) in enumerate(bold_data.items()):
            # Get vertex-level corticocortical connectivity matrix
            denoise = True
            if desc == "raw":
                denoise = False
            corr_data = bold_run.load_correlation_matrix(denoise=denoise, **denoise_settings)

            # Z-score normalize
            _bold_data = (_bold_data - _bold_data.mean(1, keepdims=True)) / _bold_data.std(1, keepdims=True)

            """1. Sample vertex timeseries
            """
            # Extract vertex timeseries
            """
            vtx_bold = _bold_data[vtx_id, :]
            n_tps = vtx_bold.shape[-1]
            TR = bold_run.repetition_time
            # Get axes' coordinates
            tps = np.arange(TR*(n_start_volumes_to_remove+1), n_tps * TR + ((n_start_volumes_to_remove+1) * TR), TR)
            frequencies, vtx_psd = welch(vtx_bold, fs=1/TR, nperseg=256)
            # Plot
            if bold_ix == 0: # raw
                c='k'
            else: # denoised:
                c='r'
            """

            """
            vtx_axs[0].plot(tps, vtx_bold, lw=.75, c=c, zorder=bold_ix+1)
            vtx_axs[1].plot(frequencies, vtx_psd, lw=.75, c=c, zorder=bold_ix+1)
            """

            """2. Carpet plot
            """
            # Plot
            if bold_ix == 0:
                mc_summary_metrics = normalize_motion_summary_metrics(mc_summary_df, n_start_volumes_to_remove)
                for metric_label, metric_ts in mc_summary_metrics.items():
                    metric_max, metric_median, norm_metric_median, metric_ts = min_max_normalize(metric_ts)
                    """
                    c = 'k'
                    if metric_label == "framewise_displacement":
                        c = 'purple'
                    """
                """
                cp_axs[0,0].text(int(len(tps)*.80), 1.1, f"Median [Max] FD: {metric_median:.3f} [{metric_max:.3f}]", fontsize=FONTSIZE,zorder=10)
                cp_axs[0,0].plot(tps, metric_ts, c=c, lw=.75)
                cp_axs[0,0].axhline(y=norm_metric_median, color='r', zorder=10, linestyle='-', lw=.5)
                """

            """
            cp_ax = cp_axs[bold_ix+1,0]
            cp_ax.imshow(
                _bold_data,
                cmap=cp_cmap,
                aspect='auto',
                vmin=cp_vmin,
                vmax=cp_vmax,
            )
            cp_ax = cp_axs[bold_ix+1,1]
            cp_ax.imshow(
                corr_data, cmap="bwr", aspect='auto', vmin=corr_vmin, vmax=corr_vmax,
            )
            """

        """Global plotting rules
        """
        """
        # Carpet plot
        for i in range(len(cp_axs)):
            for j in range(len(cp_axs[i])):
                ax = cp_axs[i,j]
                ax.set_xticks([])
                if i == 0 and j == 0:
                    ax.set_yticks([0, mc_summary_metrics['framewise_displacement'].max()])
                    new_yticks = ax.get_yticks()
                    updated_yticks = [f"{ticklabel:.3f}" for ticklabel in new_yticks]
                    ax.set_yticklabels(updated_yticks)
                else:
                    ax.set_yticks([])
                for k in ("top", "right", "bottom", "left"):
                    ax.spines[k].set_visible(False)
        base_info = figure_settings["BASE_INFO"]
        cp_fig.suptitle(f"{base_info}", fontsize=FONTSIZE)

        # Vertex timeseries plot
        for vtx_ax in vtx_axs:
            for i in ("top", "right", "bottom", "left"):
                vtx_ax.spines[i].set_visible(False)
            vtx_ax.set_xticks([])
            vtx_ax.set_yticks([])
        for f in [denoise_settings['low_pass'], denoise_settings['high_pass']]:
            vtx_axs[1].axvline(f, c='grey', linestyle='-', zorder=1)

        if figure_settings["TURN_OFF_VERTEX_PLOT"]:
            plt.close(vtx_fig)

        if figure_settings["TURN_OFF_CARPET_PLOT"]:
            plt.close(cp_fig)

        if figure_settings["SAVE_CARPET_PLOT"]:
            cp_fig.savefig(figure_settings["FIGURE_DIRECTORY"] / f"{base_info}.png", dpi=figure_settings["SAVE_DPI"])
        """

        return (
            bold_data,
            corr_data,
            metric_ts,
        )
    else:
        raise ValueError(f"{data_type} is not supported.")
