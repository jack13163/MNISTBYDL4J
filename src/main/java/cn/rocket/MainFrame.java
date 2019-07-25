
package cn.rocket;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.apache.commons.lang3.ArrayUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class MainFrame extends JFrame {
	private static final long serialVersionUID = 1L;

	private Canvas canvas;

	private MultiLayerNetwork model;
	ArrayList<?> list = new ArrayList<Object>();

	public static void main(String[] args) throws IOException {
		MainFrame drawBorder = new MainFrame();
		drawBorder.initFrame();
	}

	public void initFrame() throws IOException {
		// 图片的尺寸
		final int numRows = 28;
		final int numColumns = 28;
		final int channels = 1;

		model = ModelSerializer.restoreMultiLayerNetwork("模型.zip");

		this.setTitle("手写数字识别");
		this.setDefaultCloseOperation(3);
		this.setLocationRelativeTo(null);
		this.setResizable(false);

		JPanel panel = new JPanel();
		panel.setLayout(new BorderLayout());
		this.add(panel);

		canvas = new Canvas(680, 680);
		canvas.setPreferredSize(new Dimension(280, 280));
		this.canvas.setBounds(new Rectangle(85, 30, 280, 280));
		panel.add(canvas, BorderLayout.CENTER);

		JPanel predictPanel = new JPanel();
		predictPanel.setLayout(new BorderLayout());
		predictPanel.setBackground(Color.CYAN);
		predictPanel.setPreferredSize(new Dimension(200, 280));
		panel.add(predictPanel, BorderLayout.EAST);

		JLabel tip = new JLabel("识别结果:");
		tip.setFont(new Font("宋体", Font.BOLD, 20));
		predictPanel.add(tip, BorderLayout.NORTH);

		JLabel show = new JLabel("");
		show.setFont(new Font("宋体", Font.BOLD, 100));
		predictPanel.add(show, BorderLayout.CENTER);

		JLabel lable = new JLabel("置信度:");
		lable.setFont(new Font("宋体", Font.BOLD, 20));
		predictPanel.add(lable, BorderLayout.SOUTH);

		// 主面板添加下方面板
		JPanel paneldown = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
		paneldown.setBackground(Color.gray);
		panel.add(paneldown, BorderLayout.SOUTH);

		JLabel space = new JLabel("");
		space.setPreferredSize(new Dimension(50, 40));
		paneldown.add(space);

		JButton start = new JButton("识别");
		start.setFont(new Font("宋体", Font.BOLD, 20));
		start.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				try {

					int[] outline = getOutline();
					System.out.print(ArrayUtils.toString(outline));

					// 转换为4维行向量
					NativeImageLoader loader = new NativeImageLoader(numRows, numColumns, channels);
					INDArray image = loader.asMatrix(saveJPanel(outline));

					// 标准化为0-1之间
					DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
					scaler.transform(image);

					// 输入到模型中，进行前向计算
					INDArray output = model.output(image);

					// 找到最大的输出节点，作为识别结果
					int index = 0;
					double max = 0.000001;
					for (int i = 0; i < output.columns(); i++) {
						if (output.getDouble(0, i) >= max) {
							index = i;
							max = output.getDouble(0, i);
						}
					}

					show.setText(index + "");
					lable.setText("置信度:" + max * 100 + "%");

				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		paneldown.add(start);

		JLabel space1 = new JLabel("");
		space1.setPreferredSize(new Dimension(50, 40));
		paneldown.add(space1);

		JButton clear = new JButton("清空画板");
		clear.setFont(new Font("宋体", Font.BOLD, 20));
		clear.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				list.clear();
				canvas.clear();
			}
		});
		paneldown.add(clear);

		this.setVisible(true);
		this.pack();
	}

	public int[] getOutline() {
		double[] grayMatrix = ImageUtil.getInstance().getGrayMatrixFromPanel(canvas, null);
		int[] binaryArray = ImageUtil.getInstance().transGrayToBinaryValue(grayMatrix);
		int minRow = Integer.MAX_VALUE;
		int maxRow = Integer.MIN_VALUE;
		int minCol = Integer.MAX_VALUE;
		int maxCol = Integer.MIN_VALUE;
		for (int i = 0; i < binaryArray.length; i++) {
			int row = i / 28;
			int col = i % 28;
			if (binaryArray[i] == 0) {
				if (minRow > row) {
					minRow = row;
				}
				if (maxRow < row) {
					maxRow = row;
				}
				if (minCol > col) {
					minCol = col;
				}
				if (maxCol < col) {
					maxCol = col;
				}
			}
		}
		int len = Math.max((maxCol - minCol + 1) * 10, (maxRow - minRow + 1) * 10);

		int p = 0;
		p = (len + 40 - (maxCol - minCol + 1) * 10 - 20 - 20) / 2;
		if (p < 0)
			p = 0;

		int x = minCol * 10 - 20 - p;
		int y = minRow * 10 - 20;
		int width = len + 40;
		if (x < 0 || y < 0) {
			x = minCol * 10;
			y = minRow * 10;
			width = len;
		}
		canvas.setOutLine(x, y, width, width);

		return new int[] { x, y, width, width };
	}

	public BufferedImage saveJPanel(int[] outline) {
		Dimension imageSize = this.canvas.getSize();
		BufferedImage image = new BufferedImage(imageSize.width, imageSize.height, BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics = image.createGraphics();
		this.canvas.paint(graphics);
		graphics.dispose();
		try {
			// cut
			if (outline[0] + outline[2] > canvas.getWidth()) {
				outline[2] = canvas.getWidth() - outline[0];
			}
			if (outline[1] + outline[3] > canvas.getHeight()) {
				outline[3] = canvas.getHeight() - outline[1];
			}
			image = image.getSubimage(outline[0], outline[1], outline[2], outline[3]);
			// resize
			Image smallImage = image.getScaledInstance(Constant.smallWidth, Constant.smallHeight, Image.SCALE_SMOOTH);
			BufferedImage bSmallImage = new BufferedImage(Constant.smallWidth, Constant.smallHeight,
					BufferedImage.TYPE_INT_RGB);
			Graphics graphics1 = bSmallImage.getGraphics();
			graphics1.drawImage(smallImage, 0, 0, null);
			graphics1.dispose();

			return bSmallImage;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

}
