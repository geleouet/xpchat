package me.egaetan.xpchat;

import static dev.langchain4j.data.message.ChatMessageDeserializer.messagesFromJson;
import static dev.langchain4j.data.message.ChatMessageSerializer.messagesToJson;
import static dev.langchain4j.model.openai.OpenAiModelName.GPT_3_5_TURBO;
import static java.util.stream.Collectors.joining;
import static org.mapdb.Serializer.INTEGER;
import static org.mapdb.Serializer.STRING;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.eclipse.jetty.server.session.SessionHandler;
import org.jetbrains.annotations.NotNull;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.testcontainers.containers.Container.ExecResult;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.output.OutputFrame.OutputType;
import org.testcontainers.containers.wait.strategy.DockerHealthcheckWaitStrategy;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.dockerjava.api.model.DeviceRequest;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentLoader;
import dev.langchain4j.data.document.DocumentSource;
import dev.langchain4j.data.document.DocumentSplitter;
import  dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.embedding.E5SmallV2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import io.javalin.Javalin;
import io.javalin.community.ssl.SslPlugin;
import io.javalin.http.Context;
import io.javalin.http.UploadedFile;
import io.javalin.http.staticfiles.Location;
import io.javalin.websocket.WsConnectContext;


public class MyCM {
	
	private static final String WHISPER_MODEL = "whisper";
	private static final String STABLEDIFFUSION_URL_TXT2IMG = "http://127.0.0.1:7860/sdapi/v1/txt2img";
	private static final String PATH_TO_CERT = "C:\\Certbot\\live\\undefined.egaetan.me\\";
	static String OLLAMA_MODEL_NAME = "gemma";
	static String OLLAMA_DOCKER_IMAGE_NAME = "ollama/ollama";
	static Integer OLLAMA_PORT = 11434;
	static String DOCKER_LOCALAI_IMAGE_NAME = "localai/localai:v2.9.0-ffmpeg-core";
	static Integer LOCALAI_PORT = 8080;

	static GenericContainer<?> localai = new GenericContainer<>(DOCKER_LOCALAI_IMAGE_NAME)
			.withFileSystemBind("whisperModels", "/build/models")
			.withCommand("whisper-base")
			.withExposedPorts(8080);

	static GenericContainer<?> ollama = new GenericContainer<>(OLLAMA_DOCKER_IMAGE_NAME)
			.withCreateContainerCmdModifier(cmd -> {
				cmd
				.getHostConfig()
				.withDeviceRequests(
						Collections.singletonList(
								new DeviceRequest()
								.withCapabilities(Collections.singletonList(Collections.singletonList("gpu")))
								.withCount(-1)
								)
						);
			})
			.withFileSystemBind("ollama", "/root/.ollama")
			.withExposedPorts(OLLAMA_PORT);


	public static class Message {
		public String message;
		public Message() {
		}
		public Message(String message) {
			super();
			this.message = message;
		}
	}

	public static class MessageService {
		public String service;
		public MessageService() {
		}
		public MessageService(String message) {
			super();
			this.service = message;
		}
	}

	public static class MessageStop {
		public String stop;
		public MessageStop() {
		}
		public MessageStop(String message) {
			super();
			this.stop = message;
		}
	}

	public static class Whisper {
		public String text;
	}


	interface Assistant {
		@SystemMessage("You are a helpful french assistant. Répond uniquement en français, ne parle jamais en anglais. Sois précis et juste dans toutes tes réponses")
		TokenStream chat(@MemoryId String id, @UserMessage String userMessage);
	}


	public static void main(String[] args) throws UnsupportedOperationException, IOException, InterruptedException {
		ollama.start();
		ollama.followOutput(x -> System.out.println("OLLAMA>>"+x.getUtf8StringWithoutLineEnding()), OutputType.STDOUT);
		ollama.followOutput(x -> System.err.println("OLLAMA>>"+x.getUtf8StringWithoutLineEnding()), OutputType.STDERR);
		ollama.waitingFor(new DockerHealthcheckWaitStrategy());
		

		localai.setCommand("whisper-base");
		localai.start();
		localai.followOutput(x -> System.out.println("LOCALAI"+x.getUtf8StringWithoutLineEnding()), OutputType.STDOUT);
		localai.followOutput(x -> System.err.println("LOCALAI"+x.getUtf8StringWithoutLineEnding()), OutputType.STDERR);
		localai.waitingFor(new DockerHealthcheckWaitStrategy());

		System.out.println("Run Ollama");
		ExecResult execInContainer = ollama.execInContainer("ollama",  "run", "gemma:7b");
		System.err.println(execInContainer.getStderr());
		System.out.println(execInContainer.getStdout());

		System.out.println("Create LanguageModels");
		StreamingChatLanguageModel modelStreaming = OllamaStreamingChatModel.builder()
				.baseUrl(String.format("http://%s:%d", ollama.getHost(), ollama.getMappedPort(OLLAMA_PORT)))
				.timeout(Duration.ofMinutes(2))
				.modelName("gemma:7b")
				.numPredict(8192)
				.temperature(0.0).build();

		PersistentChatMemoryStore store = new PersistentChatMemoryStore();

		DocumentSplitter splitter = DocumentSplitters.recursive(300, 50, new OpenAiTokenizer(GPT_3_5_TURBO));
		EmbeddingModel embeddingModel = new E5SmallV2EmbeddingModel();
		Map<String, EmbeddingStore<TextSegment>> embeddingStore = new ConcurrentHashMap<>();

		ChatMemoryProvider chatMemoryProvider = memoryId -> MessageWindowChatMemory.builder().id(memoryId)
				.maxMessages(20)
				.chatMemoryStore(store)
				.build();

		Assistant assistant = AiServices.builder(Assistant.class)
				.streamingChatLanguageModel(modelStreaming)
				.chatMemoryProvider(chatMemoryProvider)
				.build();

		SslPlugin plugin = new SslPlugin(conf -> {
			conf.pemFromPath(PATH_TO_CERT + "cert.pem",
					PATH_TO_CERT + "privkey.pem");
			conf.http2 = false;
		});


		Javalin app = Javalin.create(config -> {
			config.staticFiles.add("src/main/resources/public", Location.EXTERNAL);
			config.jetty.modifyServletContextHandler(handler -> handler.setSessionHandler(new SessionHandler()));
			config.registerPlugin(plugin);
		})
				;
		app.before(ctx -> {
			ctx.req().getSession(true);
		});

		Map<String, WsConnectContext> rsp = new ConcurrentHashMap<>();
		ExecutorService executor = Executors.newFixedThreadPool(2);

		app.post("/api/chat2Img", ctx -> {
			Message msg = ctx.bodyAsClass(Message.class);
			String sessionId = ctx.req().getSession().getId();
			draw(ctx, msg, sessionId);
		});

		app.post("/api/speech", ctx -> {
			UploadedFile uploadedFile = ctx.uploadedFile("file");
			MultiPartBodyPublisher publisher = new MultiPartBodyPublisher()
					.addPart("model", WHISPER_MODEL)
					.addPart("file", () -> uploadedFile.content(), "speech", "application/octet-stream");

			HttpClient client = HttpClient.newHttpClient();
			HttpRequest request = HttpRequest.newBuilder()
					.uri(URI.create("http://localhost:"+localai.getMappedPort(LOCALAI_PORT)+"/v1/audio/transcriptions"))
					.header("Content-Type", "multipart/form-data; boundary=" + publisher.getBoundary())
					.timeout(Duration.ofMinutes(1))
					.POST(publisher.build())
					.build();

			HttpResponse<String> response = client.send(request, BodyHandlers.ofString());

			System.out.println(response.statusCode());
			System.out.println(response.body());

			ObjectMapper mapperWhisper = new ObjectMapper();
			mapperWhisper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

			Whisper value = mapperWhisper.readValue(response.body(), Whisper.class);

			Message msg = new Message(value.text);
			String sessionId = ctx.req().getSession().getId();
			System.out.println("SessionId : " + sessionId);
			generateChat(embeddingModel, embeddingStore, assistant, rsp, executor, msg, sessionId);
			ctx.json(msg);
		});
		app.post("/api/chat", ctx -> {
			Message msg = ctx.bodyAsClass(Message.class);
			String sessionId = ctx.req().getSession().getId();
			System.out.println("SessionId : " + sessionId);

			generateChat(embeddingModel, embeddingStore, assistant, rsp, executor, msg, sessionId);

			System.out.println(msg.message);
		});

		app.post("/api/upload", ctx -> {
			String sessionId = ctx.req().getSession().getId();
			System.out.println("Upload");

			UploadedFile uploadedFile = ctx.uploadedFile("document");
			InputStream content = uploadedFile.content();
			Document document = DocumentLoader.load(new DocumentSource() {

				@Override
				public Metadata metadata() {
					return new Metadata();
				}

				@Override
				public InputStream inputStream() throws IOException {
					return content;
				}
			}, new ApachePdfBoxDocumentParser());

			List<TextSegment> segments = splitter.split(document);
			List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
			embeddingStore.computeIfAbsent(sessionId, __ -> new InMemoryEmbeddingStore<>()).addAll(embeddings, segments);
			System.out.println("OK -pdf");
		});

		app.ws("/api/canal", ctx -> {
			ctx.onConnect(r -> {
				String sessionId = r.getUpgradeCtx$javalin().req().getSession().getId();
				System.out.println("Session " + sessionId);
				rsp.put(sessionId, r);
				r.sendPing();
			});
			ctx.onClose(r -> {
				String sessionId = r.getUpgradeCtx$javalin().req().getSession().getId();
				System.out.println("Delete Session " + sessionId);
				store.deleteMessages(sessionId);
				embeddingStore.remove(sessionId);
				rsp.remove(sessionId);
			});
		});

		Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(() -> {
			for (var x : rsp.values()) {
				x.sendPing();
			}
		}, 1, 1, TimeUnit.SECONDS);
		app.start(7070);

	}

	private static void draw(@NotNull Context ctx, Message msg, String sessionId)
			throws IOException, InterruptedException {
		System.out.println("Chat2img : " + msg.message);
		System.out.println("SessionId : " + sessionId);
		HttpClient client = HttpClient.newHttpClient();
		
		String body = """
				{
				"prompt": "$$$PROMPT$$$",
				"negative_prompt" : "ugly, bad quality",
				"steps": 20,
				"cfg_scale": 5,
				"sampler_name": "DPM++ 3M SDE Karras",
				"width": 512,
				"height": 512,
				"override_settings": {
				"sd_model_checkpoint": "icbinpICantBelieveIts_newYear",
				"CLIP_stop_at_last_layers": 2
				},
				"extra_generation_params": {"ADetailer model": "face_yolov8n.pt", "ADetailer confidence": 0.3, "ADetailer dilate erode": 4, "ADetailer mask blur": 4, "ADetailer denoising strength": 0.4, "ADetailer inpaint only masked": true, "ADetailer inpaint padding": 32, "ADetailer version": "24.1.2", "Denoising strength": 0.4, "Mask blur": 4, "Inpaint area": "Only masked", "Masked area padding": 32}
				}
				""";
		HttpRequest req = HttpRequest.newBuilder()
				.uri(URI.create(STABLEDIFFUSION_URL_TXT2IMG))
				.POST(BodyPublishers.ofString(body.replace("$$$PROMPT$$$", msg.message)))
				.build();

		HttpResponse<String> reponse = client.send(req, BodyHandlers.ofString());
		System.out.println("Done");
		ctx.result(reponse.body());
	}

	private static void generateChat(EmbeddingModel embeddingModel,
			Map<String, EmbeddingStore<TextSegment>> embeddingStore, Assistant assistant,
			Map<String, WsConnectContext> rsp, ExecutorService executor, Message msg, String sessionId) {

		System.out.println(">>>" + msg.message);
		EmbeddingStore<TextSegment> embeddings = embeddingStore.get(sessionId);
		if (embeddings == null) {
			executor.execute(() -> speak(assistant, rsp, msg, sessionId));
		}
		else {
			Embedding questionEmbedding = embeddingModel.embed(msg.message).content();
			int maxResults = 10;
			double minScore = 0.7;
			List<EmbeddingMatch<TextSegment>> relevantEmbeddings = embeddings.findRelevant(questionEmbedding,
					maxResults, minScore);

			PromptTemplate promptTemplate = PromptTemplate
					.from("Répond à la question suivante avec la plus grande précisions:\n" + "\n" + "Question:\n"
							+ "{{question}}\n" + "\n" + "En te basant sur les informations suivantes:\n"
							+ "{{information}}");

			String information = relevantEmbeddings.stream().map(match -> match.embedded().text())
					.collect(joining("\n\n"));
			System.out.println("Embeddings:" + information.length() +"\n------------------\n");

			Map<String, Object> variables = new HashMap<>();
			variables.put("question", msg.message);
			variables.put("information", information);

			Prompt prompt = promptTemplate.apply(variables);

			executor.execute(() -> speak(assistant, rsp, new Message(prompt.text()), sessionId));

		}
	}

	private static void speak(Assistant assistant, Map<String, WsConnectContext> rsp, Message msg, String sessionId) {
		TokenStream tokenStream = assistant.chat(sessionId, msg.message);
		AtomicBoolean receive = new AtomicBoolean(false);
		tokenStream.onNext(t -> {
			WsConnectContext x = rsp.get(sessionId);
			if (x == null) {
				System.out.println("No session");
				tokenStream.onNext(__ -> {});
				return;
			}
			try {
				x.send(new ObjectMapper().writeValueAsString(new Message(t)));
			} catch (JsonProcessingException e) {
				e.printStackTrace();
			}
			if (!receive.getAndSet(true)) {
				System.out.println("Début de la réponse");
			}
		})
		.onComplete(t -> {
			WsConnectContext x = rsp.get(sessionId);
			if (x == null) {
				return;
			}
			try {
				x.send(new ObjectMapper().writeValueAsString(new MessageService(t.content().text())));
			} catch (JsonProcessingException e) {
				e.printStackTrace();
			}
			System.out.println(t);
		})
		.onError(t-> {
			WsConnectContext x = rsp.get(sessionId);
			if (x == null) {
				return;
			}
			try {
				x.send(new ObjectMapper().writeValueAsString(new MessageStop("ERROR")));
			} catch (JsonProcessingException e) {
				e.printStackTrace();
			}
			System.err.println(t);
		})
		.start();
	}

	static class PersistentChatMemoryStore implements ChatMemoryStore {

		private final DB db = DBMaker.fileDB("multi-user-chat-memory.db").transactionEnable().make();
		private final Map<Integer, String> map = db.hashMap("messages", INTEGER, STRING).createOrOpen();

		public PersistentChatMemoryStore() {
			map.clear();
		}

		@Override
		public List<ChatMessage> getMessages(Object memoryId) {
			String json = map.get((int) memoryId.hashCode());
			return messagesFromJson(json);
		}

		@Override
		public void updateMessages(Object memoryId, List<ChatMessage> messages) {
			String json = messagesToJson(messages);
			map.put((int) memoryId.hashCode(), json);
			db.commit();
		}

		@Override
		public void deleteMessages(Object memoryId) {
			map.remove((int) memoryId.hashCode());
			db.commit();
		}
	}
}
